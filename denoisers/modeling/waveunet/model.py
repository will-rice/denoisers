"""Wave UNet Model."""
from typing import Any, Dict, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import Tensor, nn
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
)
from transformers import PreTrainedModel

from denoisers.data.waveunet import Batch
from denoisers.metrics import calculate_pesq
from denoisers.modeling.modules import Activation, DownsampleBlock1D, UpsampleBlock1D
from denoisers.modeling.waveunet.config import WaveUNetConfig
from denoisers.utils import log_audio_batch, plot_image_from_audio


class WaveUNetLightningModule(LightningModule):
    """WaveUNet Model."""

    def __init__(self, config: WaveUNetConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = WaveUNetModel(self.config)
        self.loss_fn = nn.L1Loss()
        self.snr = ScaleInvariantSignalNoiseRatio()
        self.sdr = ScaleInvariantSignalDistortionRatio()
        self.autoencoder = self.config.autoencoder
        self.last_val_batch: Any = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        return self.model(inputs)

    def training_step(
        self, batch: Batch, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        outputs = self(batch.noisy)

        if self.autoencoder:
            loss = self.loss_fn(outputs.logits, batch.audio)
        else:
            loss = self.loss_fn(outputs.noise, batch.noisy - batch.audio)

        snr = self.snr(outputs.logits, batch.audio)
        sdr = self.sdr(outputs.logits, batch.audio)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_snr", snr)
        self.log("train_sdr", sdr)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        outputs = self(batch.noisy)

        if self.autoencoder:
            loss = self.loss_fn(outputs.logits, batch.audio)
        else:
            loss = self.loss_fn(outputs.noise, batch.noisy - batch.audio)

        snr = self.snr(outputs.logits, batch.audio)
        sdr = self.sdr(outputs.logits, batch.audio)
        pesq = calculate_pesq(outputs.logits, batch.audio, self.config.sample_rate)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_snr", snr)
        self.log("val_sdr", sdr)
        self.log("pesq", pesq)

        self.last_val_batch = {
            "outputs": (
                batch.audio.detach(),
                batch.noisy.detach(),
                outputs.logits.detach(),
                batch.lengths.detach(),
            )
        }

        return loss

    def on_validation_epoch_end(self) -> None:
        """Val epoch end."""
        outputs = self.last_val_batch["outputs"]
        audio, noisy, preds, lengths = outputs
        log_audio_batch(
            audio,
            noisy,
            preds,
            lengths,
            name="val",
            sample_rate=self.config.sample_rate,
        )
        plot_image_from_audio(audio, noisy, preds, lengths, "val")
        self.snr.reset()
        self.sdr.reset()

        model_name = self.trainer.default_root_dir.split("/")[-1]
        self.model.save_pretrained(self.trainer.default_root_dir + "/" + model_name)
        self.model.push_to_hub(model_name)

        garbage_collection_cuda()

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Before optimizer step."""
        self.log_dict(grad_norm(self, norm_type=1))

    def configure_optimizers(self) -> Any:
        """Set optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )

        return optimizer


class WaveUNetModelOutputs:
    """Class for holding model outputs."""

    def __init__(self, logits: Tensor, noise: Optional[Tensor] = None) -> None:
        self.logits = logits
        self.noise = noise


class WaveUNetModel(PreTrainedModel):
    """Pretrained WaveUNet Model."""

    def __init__(self, config: WaveUNetConfig):
        super().__init__(config)
        self.config = config
        self.model = WaveUNet(
            in_channels=config.in_channels,
            downsample_kernel_size=config.downsample_kernel_size,
            upsample_kernel_size=config.upsample_kernel_size,
            dropout=config.dropout,
            activation=config.activation,
        )

    def forward(self, inputs: Tensor) -> WaveUNetModelOutputs:
        """Forward Pass."""
        if self.config.autoencoder:
            logits = self.model(inputs)
            return WaveUNetModelOutputs(logits=logits)
        else:
            noise = self.model(inputs)
            denoised = inputs - noise
            return WaveUNetModelOutputs(logits=denoised, noise=noise)


class WaveUNet(nn.Module):
    """WaveUNet Model."""

    def __init__(
        self,
        in_channels: Tuple[int, ...] = (
            24,
            48,
            72,
            96,
            120,
            144,
            168,
            192,
            216,
            240,
            264,
            288,
        ),
        downsample_kernel_size: int = 15,
        upsample_kernel_size: int = 5,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(
            1,
            in_channels[0],
            kernel_size=downsample_kernel_size,
            padding=downsample_kernel_size // 2,
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownsampleBlock1D(
                    in_channels[i],
                    out_channels=in_channels[i + 1],
                    kernel_size=downsample_kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(len(in_channels) - 1)
            ]
        )
        self.middle = nn.Sequential(
            nn.Conv1d(
                in_channels[-1],
                in_channels[-1],
                kernel_size=downsample_kernel_size,
                padding=downsample_kernel_size // 2,
            ),
            nn.BatchNorm1d(in_channels[-1]),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.decoder_layers = nn.ModuleList(
            [
                UpsampleBlock1D(
                    2 * in_channels[i + 1],
                    out_channels=in_channels[i],
                    kernel_size=upsample_kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for i in reversed(range(len(in_channels) - 1))
            ]
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels[0] + 1, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = self.in_conv(inputs)

        skips = []
        for layer in self.encoder_layers:
            out = layer(out)
            skips.append(out)

        out = self.middle(out)

        for skip, layer in zip(reversed(skips), self.decoder_layers):
            out = torch.concat([out, skip], dim=1)
            out = layer(out)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.float()

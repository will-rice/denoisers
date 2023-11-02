"""UNet1D model."""
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

from denoisers.datamodules.unet1d import Batch
from denoisers.metrics import calculate_pesq
from denoisers.modeling.unet1d.config import UNet1DConfig
from denoisers.modeling.unet1d.modules import DownBlock1D, MidBlock1D, UpBlock1D
from denoisers.utils import log_audio_batch, plot_image_from_audio


class UNet1DLightningModule(LightningModule):
    """UNet1D Lightning Module."""

    def __init__(self, config: UNet1DConfig) -> None:
        super().__init__()
        self.config = config
        self.model = UNet1DModel(config)
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


class UNet1DModelOutputs:
    """Class for holding model outputs."""

    def __init__(self, logits: Tensor, noise: Optional[Tensor] = None) -> None:
        self.logits = logits
        self.noise = noise


class UNet1DModel(PreTrainedModel):
    """Pretrained UNet1D Model."""

    config_class = UNet1DConfig

    def __init__(self, config: UNet1DConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = UNet1D(
            channels=config.channels,
            kernel_size=config.kernel_size,
            num_groups=config.num_groups,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, inputs: Tensor) -> UNet1DModelOutputs:
        """Forward Pass."""
        if self.config.autoencoder:
            logits = self.model(inputs)
            return UNet1DModelOutputs(logits=logits)
        else:
            noise = self.model(inputs)
            denoised = inputs - noise
            return UNet1DModelOutputs(logits=denoised, noise=noise)


class UNet1D(nn.Module):
    """UNet1D model."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (
            32,
            64,
            96,
            128,
            160,
            192,
            224,
            256,
            288,
            320,
            352,
            384,
        ),
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(
            1,
            channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownBlock1D(
                    channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.middle = MidBlock1D(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            activation=activation,
        )
        self.decoder_layers = nn.ModuleList(
            [
                UpBlock1D(
                    channels[i + 1],
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                )
                for i in reversed(range(len(channels) - 1))
            ]
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(channels[0] + 1, 1, kernel_size=1, padding=0),
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
            out = layer(out + skip)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.float()

"""Adapted from https://github.com/milesial/Pytorch-UNet."""
from typing import Any, Optional, Union

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

from denoisers.datamodules.unet2d import Batch
from denoisers.metrics import calculate_pesq
from denoisers.modeling.unet2d.config import UNet2DConfig
from denoisers.modeling.unet2d.modules import DownBlock2D, MidBlock2D, UpBlock2D
from denoisers.utils import log_audio_batch, plot_image_from_audio


class UNet2DLightningModule(LightningModule):
    """UNet2D Lightning Module."""

    def __init__(self, config: UNet2DConfig) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(config)
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
    ) -> Union[Tensor, dict[str, Any]]:
        """Train step."""
        outputs = self(batch.noisy)

        if self.autoencoder:
            loss = self.loss_fn(outputs.audio, batch.audio)
        else:
            loss = self.loss_fn(outputs.noise, batch.noisy - batch.audio)

        snr = self.snr(outputs.audio, batch.audio)
        sdr = self.sdr(outputs.audio, batch.audio)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_snr", snr)
        self.log("train_sdr", sdr)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, dict[str, Any]]:
        """Validate step."""
        outputs = self(batch.noisy)

        if self.autoencoder:
            loss = self.loss_fn(outputs.mag_stft, batch.specs)
        else:
            loss = self.loss_fn(outputs.noise, batch.noisy - batch.specs)

        snr = self.snr(outputs.audio, batch.audio)
        sdr = self.sdr(outputs.audio, batch.audio)
        pesq = calculate_pesq(outputs.audio, batch.audio, self.config.sample_rate)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_snr", snr)
        self.log("val_sdr", sdr)
        self.log("pesq", pesq)

        self.last_val_batch = {
            "outputs": (
                batch.audio.detach(),
                batch.noisy.detach(),
                outputs.audio.detach(),
                batch.lengths.detach(),
            ),
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


class UNet2DModelOutputs:
    """Class for holding model outputs."""

    def __init__(self, audio: Tensor, noise: Optional[Tensor] = None) -> None:
        self.audio = audio
        self.noise = noise


class UNet2DModel(PreTrainedModel):
    """Pretrained UNet1D Model."""

    config_class = UNet2DConfig

    def __init__(self, config: UNet2DConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = UNet2D(
            channels=config.channels,
            kernel_size=config.kernel_size,
            num_groups=config.num_groups,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, audio: Tensor) -> UNet2DModelOutputs:
        """Forward Pass."""
        stft = torch.stft(
            audio.squeeze(1),
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            return_complex=True,
        ).unsqueeze(1)
        noisy_mag_stft = torch.abs(stft)

        if self.config.autoencoder:
            mag_stft = self.model(noisy_mag_stft)
            noise = noisy_mag_stft - mag_stft
        else:
            noise = self.model(noisy_mag_stft)
            print(noisy_mag_stft.shape, noise.shape)
            mag_stft = noisy_mag_stft - noise

        phase = torch.angle(stft)
        zero = torch.tensor(0.0).to(mag_stft.dtype)
        phase_stft = torch.complex(mag_stft, zero) * torch.exp(
            torch.complex(zero, phase),
        )
        inv_audio = torch.istft(
            phase_stft,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
        )
        return UNet2DModelOutputs(audio=inv_audio, noise=noise)


class UNet2D(nn.Module):
    """UNet2D model."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(
            1,
            channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownBlock2D(
                    channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(len(channels) - 1)
            ],
        )
        self.middle = MidBlock2D(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            activation=activation,
        )
        self.decoder_layers = nn.ModuleList(
            [
                UpBlock2D(
                    channels[i + 1],
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                )
                for i in reversed(range(len(channels) - 1))
            ],
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1, padding=0),
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
            skip = nn.functional.pad(skip, (0, out.size(-2), 0, out.size(-1)))
            out = layer(out + skip[..., : out.shape[-2], : out.shape[-1]])

        out = self.out_conv(out)

        return out.float()

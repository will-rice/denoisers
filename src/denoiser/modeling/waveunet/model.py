"""Wave UNet Model."""
from dataclasses import dataclass
from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import SignalNoiseRatio
from diffusers import UNet1DModel

from src.denoiser import utils
from src.denoiser.datasets import Batch
from src.denoiser.utils import log_audio_batch, plot_image_from_audio


@dataclass
class WaveUNetOutputs:
    """WaveUNet outputs."""

    audio: Tensor
    noisy_audio: Tensor
    logits: Tensor


class DownSamplingLayer(nn.Module):
    """DownSampling Layer."""

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        dilation: int = 1,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 7,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.main(x)
        return x


class UpSamplingLayer(nn.Module):
    """UpSampling Layer."""

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
    ):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.main(x)
        return x


class AttentionBlock(nn.Module):
    """Attention Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_x = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_f = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Forward Pass."""
        residual = x
        x = self.conv_x(x)
        skip = self.conv_g(skip)
        x += skip
        x = self.sigmoid(x)
        x = self.conv_f(x)
        x = self.sigmoid(x)
        x *= residual
        return x


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(
        self, autoencoder: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet1DModel(in_channels=1, out_channels=1)
        self.loss_fn = nn.L1Loss()
        self.snr = SignalNoiseRatio()

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = self.unet(inputs)

        if not self.training:
            out = out.clamp(-1.0, 1.0)
            # out = AF.highpass_biquad(out, sample_rate=24000, cutoff_freq=120.0)

        return out.to(torch.float32)

    def training_step(
        self, batch: Batch, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(~masks, 0.0)

        if self.autoencoder:
            loss = self.loss_fn(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
        else:
            loss = self.loss_fn(logits, batch.noisy_audio - batch.audio)
            snr = self.snr(batch.noisy_audio - logits, batch.audio)

        self.log_dict(
            {"train_loss": loss, "train_snr": snr}, batch_size=batch.audio.size(1)
        )

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio).detach()
        logits = logits.masked_fill(~masks, 0.0)

        if self.autoencoder:
            loss = self.loss_fn(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
            pred = logits
        else:
            loss = self.loss_fn(logits, batch.noisy_audio - batch.audio)
            snr = self.snr(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict(
            {"val_loss": loss, "val_snr": snr}, batch_size=batch.audio.size(1)
        )

        return {
            "loss": loss,
            "outputs": (
                batch.audio.detach(),
                batch.noisy_audio.detach(),
                pred.detach(),
                batch.audio_lengths.detach(),
            ),
        }

    def validation_epoch_end(self, validation_step_outputs: Any) -> None:
        """Val epoch end."""
        if validation_step_outputs:
            outputs = validation_step_outputs[-1]["outputs"]
            audio, noisy, preds, lengths = outputs
            log_audio_batch(audio, noisy, preds, lengths, name="val")
            plot_image_from_audio(audio, noisy, preds, lengths, "val")

    def on_validation_epoch_end(self) -> None:
        """Val epoch end."""
        self.snr.reset()

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(~masks, 0.0)

        if self.autoencoder:
            loss = self.loss_fn(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
            pred = logits
        else:
            loss = self.loss_fn(logits, batch.noisy_audio - batch.audio)
            snr = self.snr(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict({"test_loss": loss, "test_snr": snr})

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, pred),
        }

    def configure_optimizers(self) -> Any:
        """Set optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

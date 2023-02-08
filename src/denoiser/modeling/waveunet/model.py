"""Wave UNet Model."""
import gc
from dataclasses import dataclass
from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import SignalNoiseRatio

from src.denoiser import utils
from src.denoiser.datasets import Batch
from src.denoiser.utils import log_audio_batch, plot_image_from_audio


@dataclass
class WaveUNetOutputs:
    """WaveUNet outputs."""

    audio: Tensor
    noisy_audio: Tensor
    logits: Tensor


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_groups=8
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv_in = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_out = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.residual_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.attn = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv_in(x)
        x = self.conv_out(x)
        x = x.transpose(2, 1)
        x = self.attn(x, x, x)[0]
        x = x.transpose(2, 1)
        x += residual
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, kernel_size=5, n_layers=12):
        super().__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            out_channels = 8 * (i + 1)
            self.layers.append(
                ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=2,
                )
            )
            in_channels = out_channels

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        return x, skips


class Upsample1D(nn.Module):
    """
    An upsampling layer with an optional convolution. (modified from Diffusers)
    """

    def __init__(self, in_channels, out_channels=None, use_conv_transpose=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(self.in_channels, self.out_channels, 4, 2, 1)
        else:
            self.conv = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=1)

        self.cross_attn = nn.MultiheadAttention(
            self.out_channels, num_heads=4, batch_first=True
        )

    def forward(self, x, skip):
        assert x.shape[1] == self.in_channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        x = self.cross_attn(
            x.transpose(2, 1), skip.transpose(2, 1), skip.transpose(2, 1)
        )[0].transpose(2, 1)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_layers):
        super().__init__()

        self.in_channels = in_channels
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in reversed(range(n_layers)):
            out_channels = 8 * (i + 1)
            self.layers.append(Upsample1D(in_channels, out_channels=out_channels))
            in_channels = out_channels

        self.conv_out = nn.Conv1d(out_channels, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, inputs, skips):
        out = inputs
        for skip, layer in zip(reversed(skips), self.layers):
            out = layer(out, skip)
        out = self.conv_out(out)
        out = self.tanh(out)
        return out


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(
        self, n_layers: int = 12, channels_interval: int = 24, autoencoder: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        self.autoencoder = autoencoder

        self.encoder = Encoder(1, n_layers=n_layers)
        self.middle = ResidualBlock(8 * n_layers, 8 * n_layers, kernel_size=5, stride=1)
        self.decoder = Decoder(8 * n_layers, n_layers=n_layers)

        self.loss_fn = nn.L1Loss()
        self.snr = SignalNoiseRatio()

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = inputs

        out = self.encoder(out)
        out = self.middle(out)
        out = self.decoder(out)

        if not self.training:
            out = out.clamp(-1.0, 1.0)

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
        gc.collect()
        garbage_collection_cuda()

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

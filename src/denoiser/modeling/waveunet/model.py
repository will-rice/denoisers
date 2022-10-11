"""Wave UNet Model"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import SignalNoiseRatio

from src.denoiser.data import Sample
from src.denoiser.utils import log_audio_batch


@dataclass
class WaveUNetOutputs:
    """WaveUNet outputs."""

    audio: Tensor
    noisy_audio: Tensor
    logits: Tensor


class DownSamplingBlock(nn.Module):
    """downsample and apply normalization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        dropout=0.0,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batch_norm_1 = nn.BatchNorm1d(out_channels)
        self.activation_1 = nn.LeakyReLU(0.1)
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)
        self.activation_2 = nn.LeakyReLU(0.1)
        self.max_pool = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        out = self.conv_1(inputs)
        out = self.batch_norm_1(out)
        out = self.activation_1(out)
        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.activation_2(out)
        out = self.max_pool(out)
        out = self.dropout(out)
        return out


class UpSamplingBlock(nn.Module):
    """upsample and convolve"""

    def __init__(self, in_channels, out_channels, kernel_size, padding="same"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, inputs):
        """forward pass"""
        out = self.upsample(inputs)
        out = self.conv(out)
        out = self.activation(out)
        return out


class Middle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(self, conv_sizes=(16, 32, 64, 128, 256, 512), middle_out_channels=128):
        super().__init__()
        self.conv_sizes = conv_sizes

        self.encoder_layers = nn.ModuleList()
        for i in range(len(self.conv_sizes)):
            in_channels = 1 if i == 0 else conv_sizes[i - 1]
            self.encoder_layers.append(
                DownSamplingBlock(
                    in_channels=in_channels,
                    out_channels=self.conv_sizes[i],
                    kernel_size=2**i,
                )
            )

        self.middle = Middle(
            in_channels=conv_sizes[-1], out_channels=middle_out_channels, kernel_size=16
        )

        self.decoder_layers = nn.ModuleList()
        for i in reversed(range(len(self.conv_sizes))):

            if i == len(self.conv_sizes) - 1:
                in_channels = self.conv_sizes[i] + middle_out_channels
            else:
                in_channels = self.conv_sizes[i] * 3

            self.decoder_layers.append(
                UpSamplingBlock(
                    in_channels=in_channels,
                    out_channels=self.conv_sizes[i],
                    kernel_size=2**i,
                )
            )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.conv_sizes[0],
                out_channels=1,
                kernel_size=1,
                padding="same",
            ),
            nn.Tanh(),
        )
        self.snr = SignalNoiseRatio()

    def forward(self, inputs):
        """forward pass"""
        out = inputs
        residual = out

        skip_connections = []
        for layer in self.encoder_layers:
            out = layer(out)
            skip_connections.append(out)

        out = self.middle(out)

        for layer, skip in zip(self.decoder_layers, reversed(skip_connections)):
            out = torch.cat([out, skip], axis=1)
            out = layer(out)

        out += residual
        out = self.out_conv(out)
        return out.to(torch.float32)

    def training_step(
        self, batch: Sample, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""

        logits = self(batch.noisy_audio)
        loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)

        snr = self.snr(batch.noisy_audio - logits, batch.audio)

        self.log("train_loss", loss, batch_size=batch.audio.size(1))
        self.log("train_snr", snr, batch_size=batch.audio.size(1))

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        logits = self(batch.noisy_audio)
        loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
        snr = self.snr(batch.noisy_audio - logits, batch.audio)

        self.log("val_loss", loss, batch_size=batch.audio.size(1))
        self.log("val_snr", snr, batch_size=batch.audio.size(1))

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, batch.noisy_audio - logits),
        }

    def validation_epoch_end(
        self,
        validation_step_outputs: Union[
            List[Union[Tensor, Dict[str, Any]]],
            List[List[Union[Tensor, Dict[str, Any]]]],
        ],
    ) -> None:
        audio, noisy, pred = validation_step_outputs[-1]["outputs"]
        log_audio_batch(audio, noisy, pred, name="val")

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        logits = self(batch.noisy_audio)
        loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
        snr = self.snr(batch.noisy_audio - logits, batch.audio)

        self.log("test_loss", loss, batch_size=batch.audio.size(1))
        self.log("test_snr", snr, batch_size=batch.audio.size(1))

        log_audio_batch(
            batch.audio, batch.noisy_audio, batch.noisy_audio - logits, "test"
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

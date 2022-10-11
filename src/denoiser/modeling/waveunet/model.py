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


class DownSamplingLayer(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        dilation: int = 1,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = "same",
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
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, ipt: Tensor) -> Tensor:
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = "same",
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
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(self, n_layers: int = 12, channels_interval: int = 24):
        super().__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [
            i * self.channels_interval for i in range(1, self.n_layers)
        ]
        encoder_out_channels_list = [
            i * self.channels_interval for i in range(1, self.n_layers + 1)
        ]

        # 1=>2=>3=>4=>5=>6=>7=>8=>9=>10=>11=>12
        # 16384=>8192=>4096=>2048=>1024=>512=>256=>128=>64=>32=>16=>8=>4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                )
            )
        self.pool = nn.AdaptiveAvgPool1d(2)

        self.middle = nn.Sequential(
            nn.Conv1d(
                self.n_layers * self.channels_interval,
                self.n_layers * self.channels_interval,
                15,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        decoder_in_channels_list = [
            (2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)
        ] + [2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1), nn.Tanh()
        )
        self.snr = SignalNoiseRatio()

    def forward(self, inputs: Tensor) -> Tensor:
        o = inputs

        skip_connections = []
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            skip_connections.append(o)
            # [batch_size, T // 2, channels]
            o = self.pool(o)

        o = self.middle(o)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, skip_connections[self.n_layers - i - 1]], dim=1)

            o = self.decoder[i](o)

        o = torch.cat([o, inputs], dim=1)
        o = self.out(o)
        o = o.clamp(-1.0, 1.0)
        return o.to(torch.float32)

    def training_step(
        self, batch: Sample, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""

        logits = self(batch.noisy_audio)
        loss = F.l1_loss(logits, batch.audio)

        snr = self.snr(logits, batch.audio)

        self.log("train_loss", loss, batch_size=batch.audio.size(1))
        self.log("train_snr", snr, batch_size=batch.audio.size(1))

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        logits = self(batch.noisy_audio)
        loss = F.l1_loss(logits, batch.audio)
        snr = self.snr(logits, batch.audio)

        self.log("val_loss", loss, batch_size=batch.audio.size(1))
        self.log("val_snr", snr, batch_size=batch.audio.size(1))

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, logits),
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
        loss = F.l1_loss(logits, batch.audio)
        snr = self.snr(logits, batch.audio)

        self.log("test_loss", loss, batch_size=batch.audio.size(1))
        self.log("test_snr", snr, batch_size=batch.audio.size(1))

        log_audio_batch(batch.audio, batch.noisy_audio, logits, "test")

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

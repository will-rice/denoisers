"""Wave UNet Model"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection, audio

from src.denoiser import utils
from src.denoiser.datasets.vctk import Sample
from src.denoiser.utils import log_audio_batch, plot_image_from_audio


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
        x = self.main(x)
        return x


class UpSamplingLayer(nn.Module):
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
        x = self.main(x)
        return x


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(
        self, n_layers: int = 12, channels_interval: int = 24, autoencoder=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        self.autoencoder = autoencoder

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

        self.middle = nn.Sequential(
            nn.Conv1d(
                self.n_layers * self.channels_interval,
                self.n_layers * self.channels_interval,
                15,
                stride=1,
                padding=7,
            ),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(0.2, inplace=True),
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
        metrics = MetricCollection(
            audio.SignalNoiseRatio(),
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs

        skip_connections = []
        for layer in self.encoder:
            out = layer(out)
            skip_connections.append(out)
            out = out[:, :, ::2]

        out = self.middle(out)

        # Down Sampling
        for i, layer in enumerate(self.decoder):
            # [batch_size, T * 2, channels]
            out = F.interpolate(
                out, scale_factor=2.0, mode="linear", align_corners=True
            )
            # Skip Connection
            out = torch.cat([out, skip_connections[self.n_layers - i - 1]], dim=1)
            out = layer(out)

        out = torch.cat([out, inputs], dim=1)
        out = self.out(out)

        if not self.training:
            out = out.clamp(-1.0, 1.0)

        return out.to(torch.float32)

    def training_step(
        self, batch: Sample, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            metrics = self.train_metrics(logits, batch.audio)
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
            metrics = self.train_metrics(batch.noisy_audio - logits, batch.audio)

        self.log_dict({"train_loss": loss, **metrics}, batch_size=batch.audio.size(1))

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        masks = utils.sequence_mask(
            batch.audio_lengths, batch.noisy_audio.size(-1)
        ).detach()
        logits = self(batch.noisy_audio).detach()
        logits = logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            self.val_metrics.update(logits, batch.audio)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
            self.val_metrics.update(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict({"val_loss": loss})

        return {
            "loss": loss,
            "outputs": (
                batch.audio.detach(),
                batch.noisy_audio.detach(),
                pred.detach(),
                batch.audio_lengths.detach(),
            ),
        }

    def validation_epoch_end(
        self,
        validation_step_outputs: Union[
            List[Union[Tensor, Dict[str, Any]]],
            List[List[Union[Tensor, Dict[str, Any]]]],
        ],
    ) -> None:
        audio, noisy, preds, lengths = validation_step_outputs[-1]["outputs"]
        log_audio_batch(audio, noisy, preds, lengths, name="val")
        plot_image_from_audio(audio, noisy, preds, lengths, "val")

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        masks = utils.sequence_mask(
            batch.audio_lengths, batch.noisy_audio.size(-1)
        ).detach()
        logits = self(batch.noisy_audio).detach()
        logits = logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            self.test_metrics.update(logits, batch.audio)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
            self.test_metrics.update(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict({"test_loss": loss})

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, pred),
        }

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Any:
        """Set optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9997, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_schedule}

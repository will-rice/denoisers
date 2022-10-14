"""Wave UNet Model"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import librosa as lr
import pytorch_lightning as pl
import torch
import torchaudio.functional
import torchaudio.transforms as T
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import SignalNoiseRatio

from src.denoiser import utils
from src.denoiser.data import Sample
from src.denoiser.utils import log_audio_batch, plot_image_batch


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
            nn.LeakyReLU(negative_slope=0.2),
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(
        self, n_layers: int = 12, channels_interval: int = 24, autoencoder=True
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            o = o[:, :, ::2]

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

        if not self.training:
            o = o.clamp(-1.0, 1.0)

        return o.to(torch.float32)

    def training_step(
        self, batch: Sample, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
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
        logits = self(batch.noisy_audio)
        logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
            snr = self.snr(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict(
            {"val_loss": loss, "val_snr": snr}, batch_size=batch.audio.size(1)
        )

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, pred),
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

        spectrogram = T.Spectrogram(
            n_fft=2048,
            win_length=1024,
            hop_length=256,
            center=True,
            pad_mode="constant",
            power=2.0,
        )

        original_spec = torch.from_numpy(lr.power_to_db(spectrogram(audio.to("cpu"))))
        noisy_spec = torch.from_numpy(lr.power_to_db(spectrogram(noisy.to("cpu"))))
        pred_spec = torch.from_numpy(lr.power_to_db(spectrogram(pred.to("cpu"))))

        plot_image_batch(original_spec, noisy_spec, pred_spec, "val")

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - batch.audio)
            snr = self.snr(batch.noisy_audio - logits, batch.audio)
            pred = batch.noisy_audio - logits

        self.log_dict(
            {"test_loss": loss, "test_snr": snr}, batch_size=batch.audio.size(1)
        )

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, pred),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

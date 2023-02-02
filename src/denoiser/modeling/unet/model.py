"""Adapted from https://github.com/milesial/Pytorch-UNet."""
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import functional as M

from src.denoiser.data import Batch
from src.denoiser.utils import plot_image_batch


class DoubleConv(nn.Module):
    """Convolution => [BN] => ReLU * 2."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        return self.double_conv(x)


class DownSampleLayer(nn.Module):
    """DownSample Conv Layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        return self.maxpool_conv(x)


class UpSampleLayer(nn.Module):
    """UpSample Conv Layer."""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: Optional[bool] = True
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up: Union[nn.Upsample, nn.ConvTranspose2d] = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward Pass."""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    """Output Convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        return self.conv(x)


class UNet(pl.LightningModule):
    """UNet model for image-like data NCHW."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        bilinear: Optional[bool] = True,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSampleLayer(64, 128)
        self.down2 = DownSampleLayer(128, 256)
        self.down3 = DownSampleLayer(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownSampleLayer(512, 1024 // factor)
        self.up1 = UpSampleLayer(1024, 512 // factor, bilinear)
        self.up2 = UpSampleLayer(512, 256 // factor, bilinear)
        self.up3 = UpSampleLayer(256, 128 // factor, bilinear)
        self.up4 = UpSampleLayer(128, 64, bilinear)
        self.outc = OutputConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def infer(self, audio: Tensor) -> Tensor:
        """Inference."""
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            return_complex=True,
        ).unsqueeze(1)
        mag_stft = torch.abs(stft)

        logits = self(mag_stft)
        mag_stft -= logits

        phase = torch.angle(stft)
        zero = torch.tensor(0.0).to(mag_stft.dtype)
        phase_stft = torch.complex(mag_stft, zero) * torch.exp(
            torch.complex(zero, phase)
        )
        inv_audio = torch.istft(
            phase_stft,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        return inv_audio

    def training_step(
        self, batch: Batch, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        logits = self(batch.noisy_specs)
        loss = F.l1_loss(logits, batch.noisy_specs - batch.specs)
        snr = M.signal_noise_ratio(batch.noisy_specs - logits, batch.specs)

        self.log("train_loss", loss, batch_size=batch.audio.size(1))
        self.log("train_snr", snr, batch_size=batch.audio.size(1))

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        logits = self(batch.noisy_specs)
        loss = F.l1_loss(logits, batch.noisy_specs - batch.specs)
        snr = M.signal_noise_ratio(batch.noisy_specs - logits, batch.specs)

        self.log("val_loss", loss, batch_size=batch.audio.size(1))
        self.log("val_snr", snr, batch_size=batch.audio.size(1))

        plot_image_batch(
            batch.specs, batch.noisy_specs, batch.noisy_specs - logits, "val"
        )

        return loss

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        logits = self(batch.noisy_specs)
        loss = F.l1_loss(logits, batch.noisy_specs - batch.specs)
        snr = M.signal_noise_ratio(batch.noisy_specs - logits, batch.specs)

        self.log("test_loss", loss, batch_size=batch.audio.size(1))
        self.log("test_snr", snr, batch_size=batch.audio.size(1))

        plot_image_batch(
            batch.specs, batch.noisy_specs, batch.noisy_specs - logits, "test"
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

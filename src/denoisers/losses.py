"""Denoisers losses."""

from typing import Any

import torch
from torch import nn


def stft(
    x: torch.Tensor, fft_size: int, hop_size: int, win_length: int, window: torch.Tensor
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(
        x.squeeze(1), fft_size, hop_size, win_length, window, return_complex=True
    )
    x_stft = torch.view_as_real(x_stft)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of
            predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of
            ground truth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal
            (B, frames, freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground truth signal
            (B, frames, freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: str = "hann_window",
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        window: Any = self.window
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        hop_sizes: tuple[int, ...] = (120, 240, 50),
        win_lengths: tuple[int, ...] = (600, 1200, 240),
        window: str = "hann_window",
        factor_sc: float = 0.1,
        factor_mag: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss

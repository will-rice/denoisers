"""Denoisers losses."""
import torch
from torch import Tensor, nn


class STFTLoss(nn.Module):
    """STFT Loss."""

    def __init__(
        self, n_fft: int = 2048, win_length: int = 1024, hop_length: int = 512
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.loss_fn = nn.L1Loss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Forward pass."""
        preds = torch.stft(
            preds.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        targets = torch.stft(
            targets.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        return self.loss_fn(preds, targets)

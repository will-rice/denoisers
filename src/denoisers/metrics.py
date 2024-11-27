"""Metrics for denoising."""

import torch
import torchaudio
from torch import nn
from torchmetrics.functional.audio.dnsmos import (
    deep_noise_suppression_mean_opinion_score,
)
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality


class PESQ(nn.Module):
    """PESQ metric."""

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pred_resample = self.resample(preds)
        true_resample = self.resample(target)

        shortest = min(pred_resample.shape[-1], true_resample.shape[-1])
        pred_resample = pred_resample[:, :, :shortest].squeeze(1)
        true_resample = true_resample[:, :, :shortest].squeeze(1)

        score = perceptual_evaluation_speech_quality(
            pred_resample, true_resample, fs=16000, mode="wb"
        )

        return score.mean()


class DNSMOS(nn.Module):
    """DNSMOS metric."""

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pred_resample = torchaudio.functional.resample(preds, self.sample_rate, 16000)
        score = deep_noise_suppression_mean_opinion_score(pred_resample, 16000, False)
        return score.mean()

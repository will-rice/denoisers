"""Transforms"""
import torch
from torch import Tensor, nn


class GaussianNoise(nn.Module):
    """Gaussian Noise Transform."""

    def __init__(self, min_intensity: float = 0.0, max_intensity: float = 10.0):
        super().__init__()
        self.intensity_dist = torch.distributions.uniform.Uniform(
            min_intensity, max_intensity
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        intensity = self.intensity_dist.sample().to(x.device)
        noise = torch.randn_like(x) * intensity
        noisy = x + noise
        return noisy

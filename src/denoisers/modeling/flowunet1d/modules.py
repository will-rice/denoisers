"""Modules for FlowUNet1D."""

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of the bridge time t in [0, 1]."""

    def __init__(self, embed_dim: int = 128, max_period: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps of shape (batch,) into (batch, embed_dim)."""
        half_dim = self.embed_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        freqs = torch.exp(exponent / half_dim)
        # t lives in [0, 1] so scale it up before applying frequencies.
        args = 1000.0 * timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(timesteps.dtype)


class TimeEmbedding(nn.Module):
    """Sinusoidal features followed by a small MLP."""

    def __init__(self, embed_dim: int = 512, fourier_dim: int = 128) -> None:
        super().__init__()
        self.fourier = SinusoidalTimeEmbedding(fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps of shape (batch,) into (batch, embed_dim)."""
        return self.mlp(self.fourier(timesteps))


class FiLM1D(nn.Module):
    """Feature-wise linear modulation of a (batch, channels, time) tensor."""

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * channels)
        # Zero-init so modulation starts as identity and the backbone
        # behaves like the unconditioned UNet1D at initialization.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply scale/shift conditioning to x."""
        scale, shift = self.proj(time_embedding)[:, :, None].chunk(2, dim=1)
        return x * (1.0 + scale) + shift

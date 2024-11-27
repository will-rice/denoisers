"""WaveUnet modules."""

from typing import Optional

import torch
from torch import nn

from denoisers.modeling.modules import (
    Activation,
    Downsample1D,
    Normalization,
    Upsample1D,
)


class DownsampleBlock1D(nn.Module):
    """1d downsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        bias: bool = True,
        num_groups: Optional[int] = None,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()

        self.downsample = Downsample1D(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_conv=True,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.norm = Normalization(out_channels, num_groups=num_groups, name=norm_type)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.downsample(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class UpsampleBlock1D(nn.Module):
    """1d upsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        bias: bool = True,
        norm_type: str = "batch",
        num_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.upsample = Upsample1D(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.norm = Normalization(out_channels, num_groups=num_groups, name=norm_type)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

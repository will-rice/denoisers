"""Modules for 1D U-Net."""
from torch import Tensor, nn

from denoisers.modeling.modules import (
    Activation,
    Downsample1D,
    Normalization,
    Upsample1D,
)


class DownBlock1D(nn.Module):
    """Downsampling Block for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.res_block = ResBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.downsample = Downsample1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.res_block(x)
        x = self.downsample(x)
        return x


class UpBlock1D(nn.Module):
    """Upsampling Block for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.res_block = ResBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.upsample = Upsample1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.res_block(x)
        x = self.upsample(x)
        return x


class ResBlock1D(nn.Module):
    """Residual Block for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm_1 = Normalization(in_channels, name=norm, num_groups=num_groups)
        self.activation_1 = Activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm_2 = Normalization(in_channels, name=norm, num_groups=num_groups)
        self.activation_2 = Activation(activation)
        self.residual = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        residual = self.residual(x)
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        x = self.dropout(x)
        return x + residual


class MidBlock1D(nn.Module):
    """Middle Block for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        num_heads: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.res_block_1 = ResBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.attention = nn.MultiheadAttention(out_channels, num_heads=num_heads)
        self.res_block_2 = ResBlock1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.res_block_1(x)
        residual = x
        x = x.transpose(2, 1)
        x = self.attention(x, x, x)[0]
        x = x.transpose(2, 1)
        x = x + residual
        x = self.res_block_2(x)
        return x

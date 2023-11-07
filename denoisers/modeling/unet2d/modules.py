"""UNet2d modules."""
from torch import Tensor, nn

from denoisers.modeling.modules import Activation, Downsample2D, Upsample2D


class DownBlock2D(nn.Module):
    """Downsampling Block for 2D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.res_block = ResBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.downsample = Downsample2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.res_block(x)
        x = self.downsample(x)
        return x


class UpBlock2D(nn.Module):
    """Upsampling Block for 2D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.res_block = ResBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.upsample = Upsample2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.res_block(x)
        x = self.upsample(x)
        return x


class MidBlock2D(nn.Module):
    """Middle Block for 2D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        num_heads: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.res_block_1 = ResBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(out_channels, num_heads=num_heads)
        self.res_block_2 = ResBlock2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.res_block_1(x)
        image_shape = x.size()
        x = x.reshape(image_shape[0], image_shape[1], -1)
        x = self.attention(x.transpose(2, 1), x.transpose(2, 1), x.transpose(2, 1))[
            0
        ].transpose(2, 1)
        x = x.reshape(image_shape)
        x = self.res_block_2(x)
        return x


class ResBlock2D(nn.Module):
    """Residual Block for 2D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = Activation(activation)
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.norm_2 = nn.GroupNorm(num_groups, out_channels)
        self.activation_2 = Activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        residual = self.residual(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x + residual

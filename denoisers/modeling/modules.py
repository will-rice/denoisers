"""Modules for the denoiser models."""
from typing import Any, Optional

from torch import Tensor, nn


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py#L29

    Parameters:
        in_channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        self.conv: Any = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(in_channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass."""
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = nn.functional.interpolate(
            inputs, scale_factor=2.0, mode="linear", align_corners=True
        )

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py#L70

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        kernel_size (`int`, default `3`):
            kernel size for the convolution.
        stride (`int`, default `2`):
            stride for the convolution.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        padding (`int`, default `1`):
            padding for the convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 2,
        use_conv: bool = False,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.conv: Any = None
        if use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass."""
        return self.conv(inputs)


class ResidualBlock1D(nn.Module):
    """1d residual block."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batch_norm_1 = nn.BatchNorm1d(in_channels)
        self.activation_1 = Activation(activation)
        self.dropout_1 = nn.Dropout(dropout)

        self.conv_2 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batch_norm_2 = nn.BatchNorm1d(in_channels)
        self.activation_2 = Activation(activation)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        residual = x
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = x + residual
        x = self.activation_2(x)
        x = self.dropout_2(x)
        return x


class Activation(nn.Module):
    """Activation function."""

    def __init__(self, name: str):
        super().__init__()
        if name == "silu":
            self.activation: nn.Module = nn.SiLU(inplace=True)
        elif name == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif name == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"{name} activation is not supported.")

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        x = self.activation(x)
        return x

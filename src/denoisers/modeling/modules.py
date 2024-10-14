"""Modules for the denoiser models."""

from typing import Any, Optional

import torch
from torch import nn


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py#L29

    Parameters
    ----------
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
        bias: bool = True,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        self.conv: Any = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(
                in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=bias,
            )
        elif use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = nn.functional.interpolate(
            inputs,
            scale_factor=2.0,
            mode="linear",
            align_corners=True,
        )

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py#L70

    Parameters
    ----------
        in_channels (`int`):
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
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 2,
        use_conv: bool = False,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv

        self.conv: Any = None
        if use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(inputs)


class Activation(nn.Module):
    """Activation function."""

    def __init__(self, name: str, channels: Optional[int] = None):
        super().__init__()
        if name == "silu":
            self.activation: nn.Module = nn.SiLU(inplace=True)
        elif name == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif name == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif name == "snake":
            if channels is None:
                raise ValueError(
                    "Number of channels must be specified for Snake activation."
                )
            self.activation = Snake1d(hidden_dim=channels)
        else:
            raise ValueError(f"{name} activation is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        x = self.activation(x)
        return x


class Normalization(nn.Module):
    """Normalization layer."""

    def __init__(self, in_channels: int, name: str, num_groups: Optional[int] = None):
        super().__init__()
        self.name = name
        if name == "batch":
            self.norm: nn.Module = nn.BatchNorm1d(in_channels)
        elif name == "instance":
            self.norm = nn.InstanceNorm1d(in_channels)
        elif name == "group":
            if num_groups is None:
                raise ValueError("Number of groups must be specified for GroupNorm.")
            self.norm = nn.GroupNorm(num_groups, in_channels)
        elif name == "layer":
            self.norm = nn.LayerNorm(in_channels)
        else:
            raise ValueError(f"{name} normalization is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        if self.name == "layer":
            return self.norm(x.transpose(2, 1)).transpose(2, 1)
        return self.norm(x)


class Snake1d(nn.Module):
    """A 1-dimensional Snake activation function module.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_oobleck.py#L30

    """

    def __init__(self, hidden_dim: int, logscale: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1), requires_grad=True)
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(
            alpha * hidden_states
        ).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states

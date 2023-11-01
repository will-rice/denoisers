"""Tests for modules."""
import torch

from denoisers.modeling.modules import (
    Activation,
    Downsample1D,
    DownsampleBlock1D,
    Upsample1D,
    UpsampleBlock1D,
)


def test_downsample_1d():
    """Test downsample 1d."""
    downsample = Downsample1D(1, 2, 3, 2, True, 1, True)

    assert isinstance(downsample, Downsample1D)
    assert isinstance(downsample.conv, torch.nn.Conv1d)
    assert downsample.conv.in_channels == 1
    assert downsample.conv.out_channels == 2
    assert downsample.conv.kernel_size == (3,)
    assert downsample.conv.stride == (2,)
    assert downsample.conv.padding == (1,)
    assert downsample.conv.bias is not None

    audio = torch.randn(1, 1, 800)
    out = downsample(audio)

    assert out.shape == (1, 2, 400)


def test_downsample_block_1d():
    """Test downsample block 1d."""
    block = DownsampleBlock1D(1, 2, 3, 2, 0.1, "leaky_relu", True)

    assert isinstance(block, DownsampleBlock1D)
    assert isinstance(block.downsample, Downsample1D)
    assert isinstance(block.batch_norm, torch.nn.BatchNorm1d)
    assert isinstance(block.activation.activation, torch.nn.LeakyReLU)
    assert isinstance(block.dropout, torch.nn.Dropout)

    audio = torch.randn(1, 1, 800)
    out = block(audio)

    assert out.shape == (1, 2, 400)


def test_upsample_1d():
    """Test upsample 1d."""
    upsample = Upsample1D(in_channels=1, out_channels=2, kernel_size=3, use_conv=True)

    assert isinstance(upsample, Upsample1D)
    assert isinstance(upsample.conv, torch.nn.Conv1d)
    assert upsample.conv.in_channels == 1
    assert upsample.conv.out_channels == 2
    assert upsample.conv.kernel_size == (3,)
    assert upsample.conv.stride == (1,)
    assert upsample.conv.padding == (1,)
    assert upsample.conv.bias is not None

    audio = torch.randn(1, 1, 800)
    out = upsample(audio)

    assert out.shape == (1, 2, 1600)


def test_upsample_block_1d():
    """Test upsample block 1d."""
    block = UpsampleBlock1D(1, 2, 3, 0.1, "leaky_relu", True)

    assert isinstance(block, UpsampleBlock1D)
    assert isinstance(block.upsample, Upsample1D)
    assert isinstance(block.batch_norm, torch.nn.BatchNorm1d)
    assert isinstance(block.activation.activation, torch.nn.LeakyReLU)
    assert isinstance(block.dropout, torch.nn.Dropout)

    audio = torch.randn(1, 1, 800)
    out = block(audio)

    assert out.shape == (1, 2, 1600)


def test_activation():
    """Test activation."""
    activation = Activation("relu")

    assert isinstance(activation, Activation)
    assert isinstance(activation.activation, torch.nn.ReLU)

    audio = torch.randn(1, 1, 800)
    out = activation(audio)

    assert out.shape == (1, 1, 800)

    # test leaky relu
    activation = Activation("leaky_relu")

    assert isinstance(activation, Activation)
    assert isinstance(activation.activation, torch.nn.LeakyReLU)

    audio = torch.randn(1, 1, 800)
    out = activation(audio)

    assert out.shape == (1, 1, 800)

    # test silu
    activation = Activation("silu")

    assert isinstance(activation, Activation)
    assert isinstance(activation.activation, torch.nn.SiLU)

    audio = torch.randn(1, 1, 800)
    out = activation(audio)

    assert out.shape == (1, 1, 800)

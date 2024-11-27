"""Tests for WaveUNet model."""

import torch

from denoisers.modeling.modules import Downsample1D, Normalization, Upsample1D
from denoisers.modeling.waveunet.model import WaveUNetConfig, WaveUNetModel
from denoisers.modeling.waveunet.modules import DownsampleBlock1D, UpsampleBlock1D


def test_config() -> None:
    """Test config."""
    config = WaveUNetConfig(
        max_length=8192,
        sample_rate=16000,
        in_channels=(1, 2, 3, 4, 5, 6),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
        dropout=0.1,
        activation="leaky_relu",
        autoencoder=False,
    )
    assert config.max_length == 8192
    assert config.sample_rate == 16000
    assert config.in_channels == (1, 2, 3, 4, 5, 6)
    assert config.downsample_kernel_size == 3
    assert config.upsample_kernel_size == 3
    assert config.dropout == 0.1
    assert config.activation == "leaky_relu"
    assert config.autoencoder is False


def test_model() -> None:
    """Test model."""
    config = WaveUNetConfig(
        max_length=16384,
        sample_rate=16000,
        in_channels=(1, 2, 3),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
    )
    model = WaveUNetModel(config)
    model.eval()

    audio = torch.randn(1, 1, config.max_length)
    with torch.no_grad():
        recon = model(audio).audio

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape


def test_upsample_block_1d():
    """Test upsample block 1d."""
    block = UpsampleBlock1D(1, 2, 3, 0.1, "leaky_relu", True)

    assert isinstance(block, UpsampleBlock1D)
    assert isinstance(block.upsample, Upsample1D)
    assert isinstance(block.norm, Normalization)
    assert isinstance(block.activation.activation, torch.nn.LeakyReLU)
    assert isinstance(block.dropout, torch.nn.Dropout)

    audio = torch.randn(1, 1, 800)
    out = block(audio)

    assert out.shape == (1, 2, 1600)


def test_downsample_block_1d():
    """Test downsample block 1d."""
    block = DownsampleBlock1D(1, 2, 3, 2, 0.1, "leaky_relu", True)

    assert isinstance(block, DownsampleBlock1D)
    assert isinstance(block.downsample, Downsample1D)
    assert isinstance(block.norm, Normalization)
    assert isinstance(block.activation.activation, torch.nn.LeakyReLU)
    assert isinstance(block.dropout, torch.nn.Dropout)

    audio = torch.randn(1, 1, 800)
    out = block(audio)

    assert out.shape == (1, 2, 400)

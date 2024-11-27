"""Tests for WaveUNet model."""

import torch

from denoisers.modeling.unet1d.model import UNet1DConfig, UNet1DModel


def test_config():
    """Test config."""
    config = UNet1DConfig(
        max_length=8192,
        sample_rate=16000,
        channels=(1, 2, 3, 4, 5, 6),
        kernel_size=3,
        dropout=0.1,
        activation="silu",
        autoencoder=False,
    )
    assert config.max_length == 8192
    assert config.sample_rate == 16000
    assert config.channels == (1, 2, 3, 4, 5, 6)
    assert config.kernel_size == 3
    assert config.dropout == 0.1
    assert config.activation == "silu"
    assert config.autoencoder is False


def test_model() -> None:
    """Test model."""
    config = UNet1DConfig(
        max_length=16384,
        sample_rate=16000,
        channels=(2, 4, 6, 8),
        kernel_size=3,
        num_groups=2,
    )
    model = UNet1DModel(config)
    model.eval()

    audio = torch.randn(1, 1, config.max_length)
    with torch.no_grad():
        recon = model(audio).audio

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape

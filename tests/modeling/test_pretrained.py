"""Test HuggingFace Hub pretrained models."""

from denoisers import UNet1DModel
from denoisers.modeling import WaveUNetModel


def test_waveunet_vctk_24khz():
    """Test WaveUNet 24kHz VCTK."""
    model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-24khz")
    assert model.config.sample_rate == 24000
    assert model.config.norm_type == "batch"


def test_waveunet_vctk_48khz():
    """Test WaveUNet 48kHz VCTK."""
    model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-48khz")
    assert model.config.sample_rate == 48000
    assert model.config.norm_type == "batch"


def test_unet1d_vctk_48khz():
    """Test UNet1D 48kHz VCTK."""
    model = UNet1DModel.from_pretrained("wrice/unet1d-vctk-48khz")
    assert model.config.sample_rate == 48000
    assert model.config.norm_type == "layer"

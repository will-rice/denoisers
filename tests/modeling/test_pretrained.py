"""Test HuggingFace Hub pretrained models."""

import torch

from denoisers import UNet1DModel
from denoisers.modeling import WaveUNetModel


def test_waveunet_vctk_24khz():
    """Test WaveUNet 24kHz VCTK."""
    model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-24khz")
    model.eval()
    assert model.config.sample_rate == 24000
    assert model.config.norm_type == "batch"

    audio = torch.randn(1, 1, model.config.max_length)
    with torch.no_grad():
        output = model(audio)
    assert output.audio.shape == audio.shape


def test_waveunet_vctk_48khz():
    """Test WaveUNet 48kHz VCTK."""
    model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-48khz")
    model.eval()
    assert model.config.sample_rate == 48000
    assert model.config.norm_type == "batch"

    audio = torch.randn(1, 1, model.config.max_length)
    with torch.no_grad():
        output = model(audio)
    assert output.audio.shape == audio.shape


def test_unet1d_vctk_48khz():
    """Test UNet1D 48kHz VCTK."""
    model = UNet1DModel.from_pretrained("wrice/unet1d-vctk-48khz")
    model.eval()
    assert model.dtype == torch.float32
    assert model.config.sample_rate == 48000
    assert model.config.norm_type == "layer"

    audio = torch.randn(1, 1, model.config.max_length)
    with torch.no_grad():
        output = model(audio)
    assert output.audio.shape == audio.shape


def test_unet1d_xeno_canto_32khz():
    """Test UNet1D 32kHz Xeno-canto, a legacy-layout checkpoint."""
    model, loading_info = UNet1DModel.from_pretrained(
        "wrice/unet1d-xeno-canto-32khz", output_loading_info=True
    )
    model.eval()
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert model.config.sample_rate == 32000
    assert model.config.legacy_layout is True

    audio = torch.randn(1, 1, model.config.max_length)
    with torch.no_grad():
        output = model(audio)
    assert output.audio.shape == audio.shape

"""Tests for inference utilities."""

from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from denoisers import UNet1DConfig, UNet1DModel
from denoisers.inference import denoise_file


def test_denoise_file(tmp_path: Path) -> None:
    """Test denoising a stereo file at a different sample rate."""
    config = UNet1DConfig(
        channels=(2, 4, 6, 8), num_groups=2, max_length=8192, sample_rate=16000
    )
    model = UNet1DModel(config).eval()

    input_path = tmp_path / "noisy.wav"
    # 1.3 s of stereo 22.05 kHz audio: resampled, downmixed, and not a
    # multiple of max_length, so the final chunk is padded.
    AudioEncoder(0.1 * torch.randn(2, 28665), sample_rate=22050).to_file(input_path)
    expected = (
        AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        .get_all_samples()
        .data
    )

    output_path = tmp_path / "clean.wav"
    denoise_file(model, input_path, output_path)

    output = AudioDecoder(output_path).get_all_samples()
    assert output.sample_rate == config.sample_rate
    assert output.data.shape == expected.shape

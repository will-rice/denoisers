"""Tests for inference utilities."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from denoisers import UNet1DConfig, UNet1DModel
from denoisers.inference import denoise_file


@pytest.mark.parametrize("extension", ["wav", "mp3"])
def test_denoise_file(tmp_path: Path, extension: str) -> None:
    """Streamed denoising matches denoising the whole decoded file at once."""
    config = UNet1DConfig(
        channels=(2, 4, 6, 8), num_groups=2, max_length=8192, sample_rate=16000
    )
    model = UNet1DModel(config).eval()

    # 1.3 s of stereo 22.05 kHz audio: resampled, downmixed, and not a
    # multiple of max_length, so the final chunk is padded. MP3 streams don't
    # start at 0, which chunk ranges must account for.
    input_path = tmp_path / f"noisy.{extension}"
    AudioEncoder(0.1 * torch.randn(2, 28665), sample_rate=22050).to_file(input_path)

    output_path = tmp_path / "clean.wav"
    denoise_file(model, input_path, output_path)

    audio = (
        AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        .get_all_samples()
        .data
    )
    expected = []
    for chunk in audio.split(config.max_length, dim=-1):
        padded = nn.functional.pad(chunk, (0, config.max_length - chunk.size(-1)))
        with torch.no_grad():
            output = model(padded[None]).audio
        expected.append(output[0, :, : chunk.size(-1)].clamp(-1, 1))

    output = AudioDecoder(output_path).get_all_samples()
    assert output.sample_rate == config.sample_rate
    # One 16-bit quantization step, from writing the output as WAV.
    assert torch.allclose(output.data, torch.cat(expected, dim=-1), atol=2**-15)

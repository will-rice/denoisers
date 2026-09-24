"""Inference utilities."""

from pathlib import Path

import torch
from torch import nn
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from denoisers.modeling.unet1d.model import UNet1DModel
from denoisers.modeling.waveunet.model import WaveUNetModel


def denoise_file(
    model: UNet1DModel | WaveUNetModel,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Denoise an audio file in chunks of the model's max length.

    The input is resampled to the model's sample rate and downmixed to mono.

    Args:
        model: Pretrained denoising model.
        input_path: Audio file to denoise, in any format FFmpeg can decode.
        output_path: Where to write the denoised audio. The format follows the
            file extension.
    """
    sample_rate = model.config.sample_rate
    chunk_size = model.config.max_length

    audio = (
        AudioDecoder(input_path, sample_rate=sample_rate, num_channels=1)
        .get_all_samples()
        .data
    )

    denoised = []
    for chunk in tqdm(audio.split(chunk_size, dim=-1)):
        padded = nn.functional.pad(chunk, (0, chunk_size - chunk.size(-1)))
        with torch.no_grad():
            output = model(padded[None].to(model.device)).audio
        denoised.append(output[0, :, : chunk.size(-1)])

    AudioEncoder(
        torch.cat(denoised, dim=-1).clamp(-1, 1).cpu(), sample_rate=sample_rate
    ).to_file(output_path)

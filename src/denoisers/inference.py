"""Inference utilities."""

import math
from pathlib import Path

import torch
from torch import nn
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import Encoder
from tqdm import tqdm

from denoisers.modeling.unet1d.model import UNet1DModel
from denoisers.modeling.waveunet.model import WaveUNetModel


def denoise_file(
    model: UNet1DModel | WaveUNetModel,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Denoise an audio file, streaming it in chunks of the model's max length.

    The input is resampled to the model's sample rate and downmixed to mono.
    Only one chunk is decoded, denoised, and encoded at a time, so memory use
    does not grow with the file's length. Chunks are read by time range, which
    is sample-exact for formats TorchCodec seeks precisely, such as WAV, FLAC,
    and MP3.

    Args:
        model: Pretrained denoising model.
        input_path: Audio file to denoise, in any format FFmpeg can decode.
        output_path: Where to write the denoised audio. The format follows the
            file extension.
    """
    sample_rate = model.config.sample_rate
    chunk_size = model.config.max_length
    chunk_seconds = chunk_size / sample_rate

    decoder = AudioDecoder(input_path, sample_rate=sample_rate, num_channels=1)
    begin_seconds = decoder.metadata.begin_stream_seconds
    duration_seconds = decoder.metadata.duration_seconds
    if begin_seconds is None or duration_seconds is None:
        raise ValueError(f"{input_path} does not report its stream duration.")

    end_seconds = begin_seconds + duration_seconds

    encoder = Encoder()
    stream = encoder.add_audio(sample_rate=sample_rate, num_channels=1)
    with encoder.open_file(output_path):
        # Streams don't always start at 0, so ranges are offset by the start.
        for i in tqdm(range(math.ceil(duration_seconds / chunk_seconds))):
            start_seconds = begin_seconds + i * chunk_seconds
            chunk = decoder.get_samples_played_in_range(
                start_seconds, min(start_seconds + chunk_seconds, end_seconds)
            ).data
            padded = nn.functional.pad(chunk, (0, chunk_size - chunk.size(-1)))
            with torch.no_grad():
                output = model(padded[None].to(model.device)).audio
            stream.add_samples(output[0, :, : chunk.size(-1)].clamp(-1, 1).cpu())

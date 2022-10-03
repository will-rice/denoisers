"""Dataset classes for the denoiser."""
from typing import NamedTuple, Union

from torch import Tensor


class Batch(NamedTuple):
    """Sample object for easy access to model inputs."""

    audio: Tensor
    noisy_audio: Tensor
    audio_lengths: Tensor
    specs: Tensor
    noisy_specs: Tensor
    spec_lengths: Union[int, Tensor]

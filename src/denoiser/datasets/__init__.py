from typing import NamedTuple

from torch import Tensor


class Batch(NamedTuple):
    """Sample object for easy access to model inputs."""

    audio: Tensor
    noisy_audio: Tensor
    audio_lengths: Tensor
    specs: Tensor
    noisy_specs: Tensor
    spec_lengths: Tensor

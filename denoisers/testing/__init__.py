"""Utility functions for tests."""
import torch
from torch import Tensor


def sine_wave(frequency: float, duration: float, sample_rate: int) -> Tensor:
    """Generate a sine wave.

    Args:
        frequency: Frequency of the sine wave.
        duration: Duration of the sine wave in seconds.
        sample_rate: Sample rate of the sine wave.

    Returns:
        A torch tensor containing the sine wave.
    """
    return torch.sin(
        2 * torch.pi * torch.arange(sample_rate * duration) * frequency / sample_rate
    ).unsqueeze(0)

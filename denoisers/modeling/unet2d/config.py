"""WaveUNet configuration file."""
from typing import Any

from transformers import PretrainedConfig


class UNet2DConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `UNet2DModel`."""

    model_type = "unet2d"

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        num_groups: int = 32,
        dropout: float = 0.1,
        activation: str = "silu",
        autoencoder: bool = False,
        max_length: int = 16384 * 10,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        **kwargs: Any,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.dropout = dropout
        self.activation = activation
        self.autoencoder = autoencoder
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        super().__init__(**kwargs, max_length=max_length, sample_rate=sample_rate)

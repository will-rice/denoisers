"""WaveUNet configuration file."""
from typing import Any, Tuple

from transformers import PretrainedConfig


class WaveUNetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `WaveUNetModel`."""

    model_type = "waveunet"

    def __init__(
        self,
        in_channels: Tuple[int, ...] = (
            24,
            48,
            72,
            96,
            120,
            144,
            168,
            192,
            216,
            240,
            264,
            288,
        ),
        downsample_kernel_size: int = 15,
        upsample_kernel_size: int = 5,
        dropout: float = 0.1,
        activation: str = "leaky_relu",
        autoencoder: bool = False,
        max_length: int = 16384 * 10,
        sample_rate: int = 48000,
        **kwargs: Any,
    ) -> None:
        self.in_channels = in_channels
        self.downsample_kernel_size = downsample_kernel_size
        self.upsample_kernel_size = upsample_kernel_size
        self.dropout = dropout
        self.activation = activation
        self.autoencoder = autoencoder
        super().__init__(**kwargs, max_length=max_length, sample_rate=sample_rate)

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
        kernel_size: int = 15,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        autoencoder: bool = False,
        **kwargs: Any,
    ) -> None:
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.autoencoder = autoencoder
        super().__init__(**kwargs)

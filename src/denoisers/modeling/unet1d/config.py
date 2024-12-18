"""Unet1D configuration file."""

from typing import Any, Optional

from transformers import PretrainedConfig


class UNet1DConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `UNet1DModel`."""

    model_type = "unet1d"

    def __init__(
        self,
        channels: tuple[int, ...] = (
            32,
            64,
            96,
            128,
            160,
            192,
            224,
            256,
            288,
            320,
            352,
            384,
        ),
        kernel_size: int = 3,
        num_groups: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "silu",
        autoencoder: bool = True,
        max_length: int = 48000,
        sample_rate: int = 48000,
        norm_type: str = "layer",
        **kwargs: Any,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.dropout = dropout
        self.activation = activation
        self.autoencoder = autoencoder
        self.norm_type = norm_type
        super().__init__(**kwargs, max_length=max_length, sample_rate=sample_rate)

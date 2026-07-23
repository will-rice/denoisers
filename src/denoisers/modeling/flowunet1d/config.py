"""FlowUNet1D configuration file."""

from typing import Any, Optional

from transformers import PretrainedConfig


class FlowUNet1DConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `FlowUNet1DModel`."""

    model_type = "flowunet1d"

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
        max_length: int = 48000,
        sample_rate: int = 48000,
        norm_type: str = "layer",
        time_embed_dim: int = 512,
        sigma_max: float = 0.1,
        num_inference_steps: int = 4,
        stochastic_sampling: bool = False,
        **kwargs: Any,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.dropout = dropout
        self.activation = activation
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.norm_type = norm_type
        self.time_embed_dim = time_embed_dim
        self.sigma_max = sigma_max
        self.num_inference_steps = num_inference_steps
        self.stochastic_sampling = stochastic_sampling
        super().__init__(**kwargs)

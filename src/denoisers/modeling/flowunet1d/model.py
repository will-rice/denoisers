"""FlowUNet1D model.

A waveform-domain bridge flow-matching denoiser. Training corrupts the clean
waveform along a Brownian-bridge path between the clean and noisy signals and
the network learns to predict the clean endpoint (data prediction). Sampling
starts at the noisy waveform and iteratively re-projects onto the bridge, so
a single step reduces to a predictive denoiser while more steps buy
generative refinement.
"""

from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from denoisers.modeling.flowunet1d.config import FlowUNet1DConfig
from denoisers.modeling.flowunet1d.modules import FiLM1D, TimeEmbedding
from denoisers.modeling.unet1d.modules import DownBlock1D, MidBlock1D, UpBlock1D


class FlowUNet1DModelOutputs:
    """Class for holding model outputs."""

    def __init__(self, audio: Tensor) -> None:
        self.audio = audio


class FlowUNet1DModel(PreTrainedModel):
    """Pretrained FlowUNet1D Model."""

    config_class: Any = FlowUNet1DConfig

    def __init__(self, config: FlowUNet1DConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = FlowUNet1D(
            channels=config.channels,
            kernel_size=config.kernel_size,
            num_groups=config.num_groups,
            activation=config.activation,
            dropout=config.dropout,
            norm_type=config.norm_type,
            time_embed_dim=config.time_embed_dim,
        )
        self.post_init()

    def bridge_sigma(self, timesteps: Tensor) -> Tensor:
        """Brownian-bridge noise scale with exact endpoints at t=0 and t=1."""
        return self.config.sigma_max * torch.sqrt(
            torch.clamp(timesteps * (1.0 - timesteps), min=0.0)
        )

    def predict_clean(self, sample: Tensor, noisy: Tensor, timesteps: Tensor) -> Tensor:
        """Predict the clean waveform from a bridge state at the given times."""
        return self.model(sample, noisy, timesteps)

    def forward(
        self,
        inputs: Tensor,
        num_steps: Optional[int] = None,
        stochastic: Optional[bool] = None,
    ) -> FlowUNet1DModelOutputs:
        """Denoise a noisy waveform by sampling the bridge from t=1 to t=0."""
        num_steps = num_steps or self.config.num_inference_steps
        if stochastic is None:
            stochastic = self.config.stochastic_sampling

        noisy = inputs
        sample = noisy
        batch_size = inputs.shape[0]
        denoised = noisy

        for step in range(num_steps, 0, -1):
            t = torch.full(
                (batch_size,),
                step / num_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            denoised = self.predict_clean(sample, noisy, t)
            t_prev = (step - 1) / num_steps
            sample = (1.0 - t_prev) * denoised + t_prev * noisy
            if stochastic and step > 1:
                t_prev_tensor = torch.full_like(t, t_prev)
                sigma = self.bridge_sigma(t_prev_tensor)[:, None, None]
                sample = sample + sigma * torch.randn_like(sample)

        return FlowUNet1DModelOutputs(audio=denoised)


class FlowUNet1D(nn.Module):
    """Time-conditioned 1D U-Net predicting the clean waveform."""

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
        activation: str = "silu",
        dropout: float = 0.1,
        norm_type: str = "layer",
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.in_conv = nn.Conv1d(
            2,
            channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownBlock1D(
                    channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                )
                for i in range(len(channels) - 1)
            ],
        )
        self.encoder_films = nn.ModuleList(
            [
                FiLM1D(time_embed_dim, channels[i + 1])
                for i in range(len(channels) - 1)
            ],
        )
        self.middle = MidBlock1D(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
        )
        self.middle_film = FiLM1D(time_embed_dim, channels[-1])
        self.decoder_layers = nn.ModuleList(
            [
                UpBlock1D(
                    channels[i + 1],
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                )
                for i in reversed(range(len(channels) - 1))
            ],
        )
        self.decoder_films = nn.ModuleList(
            [
                FiLM1D(time_embed_dim, channels[i])
                for i in reversed(range(len(channels) - 1))
            ],
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(channels[0] + 2, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, sample: Tensor, noisy: Tensor, timesteps: Tensor) -> Tensor:
        """Forward Pass."""
        time_embedding = self.time_embedding(timesteps)
        inputs = torch.concat([sample, noisy], dim=1)
        out = self.in_conv(inputs)

        skips = []
        for layer, film in zip(self.encoder_layers, self.encoder_films, strict=True):
            out = film(layer(out), time_embedding)
            skips.append(out)

        out = self.middle_film(self.middle(out), time_embedding)

        for skip, layer, film in zip(
            reversed(skips), self.decoder_layers, self.decoder_films, strict=False
        ):
            out = film(layer(out[..., : skip.size(-1)] + skip), time_embedding)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.float()

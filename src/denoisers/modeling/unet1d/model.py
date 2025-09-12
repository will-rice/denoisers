"""UNet1D model."""

from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from denoisers.modeling.unet1d.config import UNet1DConfig
from denoisers.modeling.unet1d.modules import DownBlock1D, MidBlock1D, UpBlock1D


class UNet1DModelOutputs:
    """Class for holding model outputs."""

    def __init__(self, audio: Tensor, noise: Optional[Tensor] = None) -> None:
        self.audio = audio
        self.noise = noise


class UNet1DModel(PreTrainedModel):
    """Pretrained UNet1D Model."""

    config_class: Any = UNet1DConfig

    def __init__(self, config: UNet1DConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = UNet1D(
            channels=config.channels,
            kernel_size=config.kernel_size,
            num_groups=config.num_groups,
            activation=config.activation,
            dropout=config.dropout,
            norm_type=config.norm_type,
        )

    def forward(self, inputs: Tensor) -> UNet1DModelOutputs:
        """Forward Pass."""
        if self.config.autoencoder:
            audio = self.model(inputs)
            return UNet1DModelOutputs(audio=audio)
        else:
            noise = self.model(inputs)
            denoised = inputs - noise
            return UNet1DModelOutputs(audio=denoised, noise=noise)


class UNet1D(nn.Module):
    """UNet1D model."""

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
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(
            1,
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
        self.middle = MidBlock1D(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
        )
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
        self.out_conv = nn.Sequential(
            nn.Conv1d(channels[0] + 1, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = self.in_conv(inputs)

        skips = []
        for layer in self.encoder_layers:
            out = layer(out)
            skips.append(out)

        out = self.middle(out)

        for skip, layer in zip(reversed(skips), self.decoder_layers):
            out = layer(out[..., : skip.size(-1)] + skip)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.float()

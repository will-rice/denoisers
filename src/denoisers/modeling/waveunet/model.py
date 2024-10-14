"""WaveUNet Model."""

from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from denoisers.modeling.modules import Activation, Normalization
from denoisers.modeling.waveunet.config import WaveUNetConfig
from denoisers.modeling.waveunet.modules import DownsampleBlock1D, UpsampleBlock1D


class WaveUNetModelOutputs:
    """Class for holding model outputs."""

    def __init__(
        self, audio: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> None:
        self.audio = audio
        self.noise = noise


class WaveUNetModel(PreTrainedModel):
    """Pretrained WaveUNet Model."""

    config_class = WaveUNetConfig

    def __init__(self, config: WaveUNetConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = WaveUNet(
            in_channels=config.in_channels,
            downsample_kernel_size=config.downsample_kernel_size,
            upsample_kernel_size=config.upsample_kernel_size,
            dropout=config.dropout,
            activation=config.activation,
        )

    def forward(self, inputs: torch.Tensor) -> WaveUNetModelOutputs:
        """Forward Pass."""
        if self.config.autoencoder:
            audio = self.model(inputs)
            return WaveUNetModelOutputs(audio=audio)
        else:
            noise = self.model(inputs)
            denoised = inputs - noise
            return WaveUNetModelOutputs(audio=denoised, noise=noise)


class WaveUNet(nn.Module):
    """WaveUNet Model."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = (
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
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        norm_type: str = "batch",
        num_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(
            1,
            in_channels[0],
            kernel_size=downsample_kernel_size,
            padding=downsample_kernel_size // 2,
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownsampleBlock1D(
                    in_channels[i],
                    out_channels=in_channels[i + 1],
                    kernel_size=downsample_kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
                for i in range(len(in_channels) - 1)
            ],
        )
        self.middle = nn.Sequential(
            nn.Conv1d(
                in_channels[-1],
                in_channels[-1],
                kernel_size=downsample_kernel_size,
                padding=downsample_kernel_size // 2,
            ),
            Normalization(in_channels[-1], num_groups=num_groups, name=norm_type),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.decoder_layers = nn.ModuleList(
            [
                UpsampleBlock1D(
                    2 * in_channels[i + 1],
                    out_channels=in_channels[i],
                    kernel_size=upsample_kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for i in reversed(range(len(in_channels) - 1))
            ],
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels[0] + 1, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        out = self.in_conv(inputs)

        skips = []
        for layer in self.encoder_layers:
            out = layer(out)
            skips.append(out)

        out = self.middle(out)

        for skip, layer in zip(reversed(skips), self.decoder_layers):
            out = torch.concat([out[..., : skip.size(-1)], skip], dim=1)
            out = layer(out)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.float()

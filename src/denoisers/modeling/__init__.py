"""Models."""

from typing import Any

from denoisers.modeling.unet1d.config import UNet1DConfig
from denoisers.modeling.unet1d.model import UNet1DModel
from denoisers.modeling.waveunet.config import WaveUNetConfig
from denoisers.modeling.waveunet.model import WaveUNetModel

__all__ = [
    "WaveUNetConfig",
    "WaveUNetModel",
    "UNet1DConfig",
    "UNet1DModel",
    "MODELS",
    "CONFIGS",
]

MODELS: dict[str, Any] = {
    "unet1d": UNet1DModel,
    "waveunet": WaveUNetModel,
}  # Add your models here
CONFIGS: dict[str, Any] = {
    "unet1d": UNet1DConfig,
    "waveunet": WaveUNetConfig,
}  # Add your configs here

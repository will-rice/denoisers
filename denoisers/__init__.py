"""Denoisers for the 1D and 2D cases."""
from denoisers.modeling.waveunet.config import WaveUNetConfig
from denoisers.modeling.waveunet.model import WaveUNetModel

__all__ = ["WaveUNetConfig", "WaveUNetModel"]

"""Publish model script."""
import argparse
from pathlib import Path
from typing import Any

from denoisers import UNet1DConfig, UNet1DModel, WaveUNetConfig, WaveUNetModel

MODELS: dict[str, Any] = {
    "unet1d": UNet1DModel,
    "waveunet": WaveUNetModel,
}  # Add your models here
CONFIGS: dict[str, Any] = {
    "unet1d": UNet1DConfig,
    "waveunet": WaveUNetConfig,
}  # Add your configs here


def main() -> None:
    """Run publishing."""
    parser = argparse.ArgumentParser("publish parser")
    parser.add_argument("model", type=str, choices=MODELS.keys())
    parser.add_argument("name", type=str)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    model = MODELS[args.model](CONFIGS[args.model]())
    model.load_pretrained(args.path)
    model.push_to_hub(args.name)


if __name__ == "__main__":
    main()

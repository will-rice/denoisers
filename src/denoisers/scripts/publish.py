"""Publish model script."""

import argparse
from pathlib import Path

from denoisers.modeling import CONFIGS, MODELS


def main() -> None:
    """Run publish."""
    parser = argparse.ArgumentParser("Publish model to huggingface hub.")
    parser.add_argument("model", type=str, choices=MODELS.keys())
    parser.add_argument("name", type=str)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    model = MODELS[args.model](CONFIGS[args.model]())
    model.from_pretrained(args.path)
    model.push_to_hub(args.name)


if __name__ == "__main__":
    main()

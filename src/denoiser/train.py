"""Train script."""
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.denoiser.data import LibriTTSDataModule
from src.denoiser.model import UNet


def main() -> None:
    """Main"""

    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("--project", default="denoiser", type=str)
    parser.add_argument("--num_devices", default="1", type=str)
    parser.add_argument("--dataset", default="libritts", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    model = UNet()
    datamodule = LibriTTSDataModule()
    logger = WandbLogger(project=args.project, log_model="all", name=args.name)

    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=1000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

"""Train script."""
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers

from src.denoiser.data import LibriTTSDataModule
from src.denoiser.modeling.unet.model import UNet


def main() -> None:
    """Main"""

    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("--project", default="denoiser", type=str)
    parser.add_argument(
        "--num_devices", default=1 if torch.cuda.is_available() else None
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--dataset", default="libritts", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    model = UNet()
    datamodule = LibriTTSDataModule(batch_size=args.batch_size)
    logger = loggers.WandbLogger(
        project=args.project,
        save_dir="logs",
        log_model=False if args.debug else "all",
        name=args.name,
        offline=args.debug,
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=1000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        val_check_interval=1000,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

"""Train script."""
import argparse
from pathlib import Path

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
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--dataset", default="libritts", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--logdir", default="logs", type=Path)
    args = parser.parse_args()

    model = UNet()
    datamodule = LibriTTSDataModule(batch_size=args.batch_size)
    logger = loggers.WandbLogger(
        project=args.project,
        save_dir=args.logdir,
        log_model=False if args.debug else "all",
        name=args.name,
        offline=args.debug,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.logdir / args.name, filename="{epoch}-{val_loss:.2f}"
    )
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=1000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        val_check_interval=1000,
        precision=16,
        callbacks=[checkpoint_callback, swa_callback],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

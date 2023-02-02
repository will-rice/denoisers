"""Train script."""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers

from src.denoiser.data import AudioFromFileDataModule
from src.denoiser.modeling.waveunet.model import WaveUNet


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--project", default="denoiser", type=str)
    parser.add_argument(
        "--num_devices", default=1 if torch.cuda.is_available() else None
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    model = WaveUNet()

    datamodule = AudioFromFileDataModule(args.data_path, batch_size=args.batch_size)

    logger = loggers.WandbLogger(
        project=args.project,
        save_dir=log_path,
        name=args.name,
        offline=args.debug,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_path,
        filename="{step}",
        save_last=True,
    )
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-6)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    pretrained = args.checkpoint_path
    last_checkpoint = pretrained if pretrained else log_path / "last.ckpt"

    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=1000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, swa_callback, lr_monitor],
        track_grad_norm=True,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )
    trainer.test(model, datamodule=datamodule)

    torch.jit.save(model.to_torchscript(), log_path / "model.pt")


if __name__ == "__main__":
    main()

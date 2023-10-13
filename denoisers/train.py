"""Train script."""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers

from denoisers.data.waveunet import AudioFromFileDataModule
from denoisers.datasets.audio import AudioDataset
from denoisers.modeling.waveunet.config import WaveUNetConfig
from denoisers.modeling.waveunet.model import WaveUNetLightningModule

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--project", default="denoisers", type=str)
    parser.add_argument(
        "--num_devices", default=1 if torch.cuda.is_available() else None
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    config = WaveUNetConfig()
    model = WaveUNetLightningModule(config)

    dataset = AudioDataset(args.data_path)
    datamodule = AudioFromFileDataModule(
        dataset,
        batch_size=args.batch_size,
        max_length=config.max_length,
        sample_rate=config.sample_rate,
    )

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
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    pretrained = args.checkpoint_path
    last_checkpoint = pretrained if pretrained else log_path / "last.ckpt"

    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=1000,
        accelerator="auto",
        val_check_interval=0.5,
        devices=args.num_devices,
        logger=logger,
        precision="16-mixed",
        accumulate_grad_batches=2,
        limit_val_batches=10,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )


if __name__ == "__main__":
    main()

"""Train script."""

import argparse
import warnings
from pathlib import Path

import torch
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything

from denoisers.datamodule import DenoisersDataModule
from denoisers.datasets.audio import AudioDataset
from denoisers.lightning_module import DenoisersLightningModule
from denoisers.modeling import CONFIGS, MODELS

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


warnings.filterwarnings("ignore")


def main() -> None:
    """Run training."""
    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("model", type=str, choices=MODELS.keys())
    parser.add_argument("name", type=str)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--project", default="denoisers", type=str)
    parser.add_argument(
        "--num_devices",
        default=1 if torch.cuda.is_available() else None,
        type=int,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    config = CONFIGS[args.model]()
    model = MODELS[args.model](config)
    lightning_module = DenoisersLightningModule(
        model,
        sync_dist=args.num_devices > 1,
        use_ema=args.ema,
        push_to_hub=args.push_to_hub,
    )

    dataset = AudioDataset(
        args.data_root, max_length=config.max_length, sample_rate=config.sample_rate
    )
    datamodule = DenoisersDataModule(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    logger = loggers.WandbLogger(
        project=args.project,
        save_dir=log_path,
        name=args.name,
        offline=args.debug,
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=log_path,
        filename="{step}",
        save_last=True,
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")

    pretrained = args.checkpoint_path
    last_checkpoint = pretrained if pretrained else log_path / "last.ckpt"

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=10000,
        accelerator="auto",
        val_check_interval=0.25 if len(dataset) // args.batch_size > 5000 else 1.0,
        devices=args.num_devices,
        logger=logger,
        precision="bf16-mixed",
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy="deepspeed_stage_2" if args.num_devices > 1 else "auto",
    )

    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )


if __name__ == "__main__":
    main()

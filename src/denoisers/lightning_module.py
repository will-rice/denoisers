"""Denoisers lightning module trainer."""

from typing import Any

import torch
import torchaudio
import wandb
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

from denoisers.datasets.audio import Batch
from denoisers.losses import MultiResolutionSTFTLoss
from denoisers.metrics import DNSMOS
from denoisers.utils import log_audio_batch, plot_image_from_audio


class DenoisersLightningModule(LightningModule):
    """Denoisers lightning module."""

    def __init__(
        self,
        model: nn.Module,
        sync_dist: bool = False,
        use_ema: bool = False,
        push_to_hub: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        if use_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
            )
        self.loss_fn = nn.L1Loss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.train_metrics = MetricCollection(
            {
                "train_snr": SignalNoiseRatio(),
                "train_sdr": SignalDistortionRatio(),
                "train_sisnr": ScaleInvariantSignalNoiseRatio(),
                "train_sisdr": ScaleInvariantSignalDistortionRatio(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_snr": SignalNoiseRatio(),
                "val_sdr": SignalDistortionRatio(),
                "val_sisnr": ScaleInvariantSignalNoiseRatio(),
                "val_sisdr": ScaleInvariantSignalDistortionRatio(),
            }
        )
        self.dns_mos = DNSMOS()
        self.autoencoder = self.model.config.autoencoder
        self.sync_dist = sync_dist
        self.use_ema = use_ema
        self.push_to_hub = push_to_hub
        self.last_val_batch: dict[str, Any] = {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        return self.model(inputs)

    def training_step(self, batch: Batch, batch_idx: Any) -> torch.Tensor:
        """Train step."""
        outputs = self(batch.noisy)

        if self.autoencoder:
            l1_loss = self.loss_fn(outputs.audio, batch.audio)
        else:
            l1_loss = self.loss_fn(outputs.noise, batch.noisy - batch.audio)

        stft_loss = self.stft_loss(outputs.audio.float(), batch.audio.float())

        loss = l1_loss + stft_loss

        metrics = self.train_metrics(outputs.audio, batch.audio)

        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log_dict(
            {**metrics, "train_stft_loss": stft_loss, "train_l1_loss": l1_loss},
            sync_dist=self.sync_dist,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: Any) -> torch.Tensor:
        """Val step."""
        if self.use_ema:
            outputs = self.ema_model(batch.noisy)
        else:
            outputs = self.model(batch.noisy)

        if self.autoencoder:
            l1_loss = self.loss_fn(outputs.audio, batch.audio)
        else:
            l1_loss = self.loss_fn(outputs.noise, batch.noisy - batch.audio)

        stft_loss = self.stft_loss(outputs.audio.float(), batch.audio.float())

        loss = l1_loss + stft_loss

        metrics = self.val_metrics(outputs.audio, batch.audio)

        self.log("val_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log_dict(
            {**metrics, "val_stft_loss": stft_loss, "val_l1_loss": l1_loss},
            sync_dist=self.sync_dist,
        )

        self.last_val_batch = {
            "outputs": (
                batch.audio.detach(),
                batch.noisy.detach(),
                outputs.audio.detach(),
                batch.lengths.detach(),
            ),
        }

        return loss

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        """Val epoch end."""
        outputs = self.last_val_batch["outputs"]
        audio, noisy, preds, lengths = outputs

        score = deep_noise_suppression_mean_opinion_score(
            torchaudio.functional.resample(
                preds.cpu(), self.model.config.sample_rate, 16000
            ),
            16000,
            False,
        ).mean()
        wandb.log({"dns_mos": score})

        log_audio_batch(
            audio,
            noisy,
            preds,
            lengths,
            name="val",
            sample_rate=self.model.config.sample_rate,
        )
        plot_image_from_audio(audio, noisy, preds, lengths, "val")

        if self.use_ema:
            self.model.load_state_dict(self.ema_model.module.state_dict())

        model_name = self.trainer.default_root_dir.split("/")[-1]
        self.model.save_pretrained(self.trainer.default_root_dir + "/" + model_name)

        if self.push_to_hub:
            self.model.push_to_hub(model_name)

        garbage_collection_cuda()

    def on_before_zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Update EMA model."""
        if self.use_ema:
            self.ema_model.update_parameters(self.model)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Before optimizer step."""
        self.log_dict(grad_norm(self, norm_type=1), sync_dist=self.sync_dist)

    def configure_optimizers(self) -> Any:
        """Set optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

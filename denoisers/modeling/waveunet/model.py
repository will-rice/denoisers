"""Wave UNet Model."""
from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import Tensor, nn
from torchmetrics.audio import SignalNoiseRatio

from denoisers import utils
from denoisers.data.waveunet import Batch
from denoisers.modeling.modules import Activation, Downsample1D, Upsample1D
from denoisers.utils import log_audio_batch, plot_image_from_audio


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(self, autoencoder: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WaveUNetModel()
        self.loss_fn = nn.L1Loss()
        self.snr = SignalNoiseRatio()
        self.autoencoder = autoencoder
        self.last_val_batch: Any = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        return self.model(inputs)

    def training_step(
        self, batch: Batch, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        masks = utils.sequence_mask(batch.lengths, batch.noisy.size(-1))
        logits = self(batch.noisy)
        logits = logits.masked_fill(~masks, 0.0)

        if self.autoencoder:
            loss = self.loss_fn(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
        else:
            loss = self.loss_fn(logits, batch.noisy - batch.audio)
            snr = self.snr(batch.noisy - logits, batch.audio)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_snr", snr)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        masks = utils.sequence_mask(batch.lengths, batch.noisy.size(-1))
        logits = self(batch.noisy)
        logits = logits.masked_fill(~masks, 0.0)

        if self.autoencoder:
            loss = self.loss_fn(logits, batch.audio)
            snr = self.snr(logits, batch.audio)
            pred = logits
        else:
            loss = self.loss_fn(logits, batch.noisy - batch.audio)
            snr = self.snr(batch.noisy - logits, batch.audio)
            pred = batch.noisy - logits

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_snr", snr)

        self.last_val_batch = {
            "outputs": (
                batch.audio.detach(),
                batch.noisy.detach(),
                pred.detach(),
                batch.lengths.detach(),
            )
        }

        return loss

    def on_validation_epoch_end(self) -> None:
        """Val epoch end."""
        outputs = self.last_val_batch["outputs"]
        audio, noisy, preds, lengths = outputs
        log_audio_batch(audio, noisy, preds, lengths, name="val")
        plot_image_from_audio(audio, noisy, preds, lengths, "val")
        self.snr.reset()

        trace = torch.jit.trace(self.model, torch.randn(1, 1, 16384 * 10).cuda())
        torch.jit.save(trace, self.trainer.default_root_dir + "/waveunet.pt")

        garbage_collection_cuda()

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Before optimizer step."""
        self.log_dict(grad_norm(self, norm_type=1))

    def configure_optimizers(self) -> Any:
        """Set optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )

        return optimizer


class WaveUNetModel(nn.Module):
    """WaveUNet Model."""

    def __init__(
        self,
        in_channels: Tuple[int, ...] = (
            24,
            48,
            72,
            96,
            120,
            144,
            168,
            192,
            216,
            240,
            264,
            288,
        ),
        kernel_size: int = 15,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(
            1, in_channels[0], kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.encoder_layers = nn.ModuleList(
            [
                DownsampleBlock1D(
                    in_channels[i],
                    out_channels=in_channels[i + 1],
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(len(in_channels) - 1)
            ]
        )
        self.middle = nn.Sequential(
            nn.Conv1d(
                in_channels[-1],
                in_channels[-1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(in_channels[-1]),
            Activation(activation),
            nn.Dropout(dropout),
        )
        self.decoder_layers = nn.ModuleList(
            [
                UpsampleBlock1D(
                    2 * in_channels[i + 1],
                    out_channels=in_channels[i],
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for i in reversed(range(len(in_channels) - 1))
            ]
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels[0] + 1, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = self.in_conv(inputs)

        skips = []
        for layer in self.encoder_layers:
            out = layer(out)
            skips.append(out)

        out = self.middle(out)

        for skip, layer in zip(reversed(skips), self.decoder_layers):
            out = torch.concat([out, skip], dim=1)
            out = layer(out)

        out = torch.concat([out, inputs], dim=1)
        out = self.out_conv(out)

        return out.to(torch.float32)


class DownsampleBlock1D(nn.Module):
    """1d downsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()

        self.downsample = Downsample1D(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_conv=True,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.downsample(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class UpsampleBlock1D(nn.Module):
    """1d upsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.upsample = Upsample1D(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_conv=True,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.upsample(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

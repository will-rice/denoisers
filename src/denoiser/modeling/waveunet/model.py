"""Wave UNet Model."""
from typing import Any, Dict, List, NamedTuple, Union

import librosa as lr
import pytorch_lightning as pl
import torch
import torchaudio.transforms as T
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import SignalNoiseRatio

from src.denoiser import utils
from src.denoiser.data import MAX_LENGTH, Sample
from src.denoiser.modeling.waveunet.layers import ConvLayer, Resample1d, centre_crop
from src.denoiser.utils import log_audio_batch, plot_image_batch


class WaveUNetOutputs(NamedTuple):
    """WaveUNet outputs."""

    audio: Tensor
    noisy_audio: Tensor
    logits: Tensor


class WaveUNet(pl.LightningModule):
    """WaveUNet Model."""

    def __init__(
        self,
        num_inputs=1,
        num_channels=(32, 64, 96, 128, 160, 192, 224, 256),
        num_outputs=1,
        kernel_size=5,
        target_output_size=MAX_LENGTH,
        conv_type="gn",
        res="fixed",
        depth=1,
        strides=2,
        autoencoder=True,
    ):
        super().__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.autoencoder = autoencoder
        # Only odd filter kernels allowed
        assert kernel_size % 2 == 1

        self.snr = SignalNoiseRatio()

        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            in_ch = num_inputs if i == 0 else num_channels[i]
            self.downsample_blocks.append(
                DownsamplingBlock(
                    in_ch,
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_size,
                    strides,
                    depth,
                    conv_type,
                    res,
                )
            )

        for i in range(0, self.num_levels - 1):
            self.upsample_blocks.append(
                UpsamplingBlock(
                    num_channels[-1 - i],
                    num_channels[-2 - i],
                    num_channels[-2 - i],
                    kernel_size,
                    strides,
                    depth,
                    conv_type,
                    res,
                )
            )

        self.bottleneck = ConvLayer(
            num_channels[-1], num_channels[-1], kernel_size, 1, conv_type
        )

        self.output_conv = nn.Conv1d(num_channels[0], num_outputs, 1)

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print(
            "Using valid convolutions with "
            + str(self.input_size)
            + " inputs and "
            + str(self.output_size)
            + " outputs"
        )

        assert (self.input_size - self.output_size) % 2 == 0
        self.shapes = {
            "output_start_frame": (self.input_size - self.output_size) // 2,
            "output_end_frame": (self.input_size - self.output_size) // 2
            + self.output_size,
            "output_frames": self.output_size,
            "input_frames": self.input_size,
        }

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsample_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert output_size >= target_output_size
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass."""
        shortcuts = []
        out = inputs

        for block in self.downsample_blocks:
            out, short = block(out)
            shortcuts.append(short)

        out = self.bottleneck(out)

        for idx, block in enumerate(self.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        out = self.output_conv(out)

        if not self.training:
            out = out.clamp(min=-1.0, max=1.0)

        return out.to(torch.float32)

    def training_step(
        self, batch: Sample, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(masks, 0.0)
        targets = batch.audio.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, targets)
            snr = self.snr(logits, targets)
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - targets)
            snr = self.snr(batch.noisy_audio - logits, targets)

        self.log_dict(
            {"train_loss": loss, "train_snr": snr}, batch_size=batch.audio.size(1)
        )

        return loss

    def validation_step(
        self, batch: Any, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Val step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(masks, 0.0)
        targets = batch.audio.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, targets)
            snr = self.snr(logits, targets)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - targets)
            snr = self.snr(batch.noisy_audio - logits, targets)
            pred = batch.noisy_audio - logits

        self.log_dict(
            {"val_loss": loss, "val_snr": snr}, batch_size=batch.audio.size(1)
        )

        audio_trimmed = [a[:l] for a, l in zip(batch.audio, batch.audio_lengths)]
        noisy_audio_trimmed = [
            n[:l] for n, l in zip(batch.noisy_audio, batch.audio_lengths)
        ]
        pred_trimmed = [p[:l] for p, l in zip(pred, batch.audio_lengths)]

        return {
            "loss": loss,
            "outputs": (audio_trimmed, noisy_audio_trimmed, pred_trimmed),
        }

    def validation_epoch_end(
        self,
        validation_step_outputs: Union[
            List[Union[Tensor, Dict[str, Any]]],
            List[List[Union[Tensor, Dict[str, Any]]]],
        ],
    ) -> None:
        audio, noisy, pred = validation_step_outputs[-1]["outputs"]
        log_audio_batch(audio, noisy, pred, name="val")

        spectrogram = T.Spectrogram(
            n_fft=2048,
            win_length=1024,
            hop_length=256,
            center=True,
            pad_mode="constant",
            power=2.0,
        )

        original_spec = torch.from_numpy(lr.power_to_db(spectrogram(audio.to("cpu"))))
        noisy_spec = torch.from_numpy(lr.power_to_db(spectrogram(noisy.to("cpu"))))
        pred_spec = torch.from_numpy(lr.power_to_db(spectrogram(pred.to("cpu"))))

        plot_image_batch(original_spec, noisy_spec, pred_spec, "val")

    def test_step(self, batch: Any, batch_idx: Any) -> Union[Tensor, Dict[str, Any]]:
        """Test step."""
        masks = utils.sequence_mask(batch.audio_lengths, batch.noisy_audio.size(-1))
        logits = self(batch.noisy_audio)
        logits = logits.masked_fill(masks, 0.0)
        targets = batch.audio.masked_fill(masks, 0.0)

        if self.autoencoder:
            loss = F.l1_loss(logits, targets)
            snr = self.snr(logits, targets)
            pred = logits
        else:
            loss = F.l1_loss(logits, batch.noisy_audio - targets)
            snr = self.snr(batch.noisy_audio - logits, targets)
            pred = batch.noisy_audio - logits

        self.log_dict(
            {"test_loss": loss, "test_snr": snr}, batch_size=batch.audio.size(1)
        )

        return {
            "loss": loss,
            "outputs": (batch.audio, batch.noisy_audio, pred),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(UpsamplingBlock, self).__init__()
        assert stride > 1

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(
                n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True
            )

        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(
                torch.cat([combined, centre_crop(upsampled, combined)], dim=1)
            )
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(DownsamplingBlock, self).__init__()
        assert stride > 1

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(
                n_outputs, 15, stride
            )  # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(
                n_outputs, n_outputs, kernel_size, stride, conv_type
            )

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

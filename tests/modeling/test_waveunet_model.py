"""Tests for WaveUNet model."""
import torch

from denoisers.datamodules.waveunet import Batch
from denoisers.modeling.waveunet.model import (
    WaveUNetConfig,
    WaveUNetLightningModule,
    WaveUNetModel,
)
from denoisers.testing import sine_wave


def test_config():
    config = WaveUNetConfig(
        max_length=8192,
        sample_rate=16000,
        in_channels=(1, 2, 3, 4, 5, 6),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
        dropout=0.1,
        activation="leaky_relu",
        autoencoder=False,
    )
    assert config.max_length == 8192
    assert config.sample_rate == 16000
    assert config.in_channels == (1, 2, 3, 4, 5, 6)
    assert config.downsample_kernel_size == 3
    assert config.upsample_kernel_size == 3
    assert config.dropout == 0.1
    assert config.activation == "leaky_relu"
    assert config.autoencoder is False


def test_model():
    """Test model."""
    config = WaveUNetConfig(
        max_length=1024,
        sample_rate=16000,
        in_channels=(1, 2, 3),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
    )
    model = WaveUNetModel(config)
    model.eval()

    audio = sine_wave(800, config.max_length, config.sample_rate)[None]
    with torch.no_grad():
        recon = model(audio).logits

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape


def test_lightning_module():
    """Test lightning module."""
    config = WaveUNetConfig(
        max_length=1024,
        sample_rate=16000,
        in_channels=(1, 2, 3),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
    )
    model = WaveUNetLightningModule(config)

    audio = sine_wave(800, config.max_length, config.sample_rate)[None]
    batch = Batch(audio=audio, noisy=audio, lengths=torch.tensor([audio.shape[-1]]))

    # test forward
    with torch.no_grad():
        recon = model(audio).logits

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape

    # test training step
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # test validation step
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

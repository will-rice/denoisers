"""Test lightning module."""

import torch

from denoisers import UNet1DConfig, UNet1DModel, WaveUNetConfig, WaveUNetModel
from denoisers.datasets.audio import Batch
from denoisers.lightning_module import DenoisersLightningModule


def test_waveunet_lightning_module() -> None:
    """Test waveunet lightning module."""
    config = WaveUNetConfig(
        max_length=16384,
        sample_rate=16000,
        in_channels=(1, 2, 3),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
    )
    model = WaveUNetModel(config)
    lightning_module = DenoisersLightningModule(model)

    audio = torch.randn(1, 1, config.max_length)
    batch = Batch(audio=audio, noisy=audio, lengths=torch.tensor([audio.shape[-1]]))

    # test forward
    with torch.no_grad():
        recon = lightning_module(audio).audio

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape

    # test training step
    loss = lightning_module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # test validation step
    loss = lightning_module.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_unet1d_lightning_module() -> None:
    """Test unet1d lightning module."""
    config = UNet1DConfig(
        max_length=16384,
        sample_rate=16000,
        in_channels=(1, 2, 3),
        downsample_kernel_size=3,
        upsample_kernel_size=3,
    )
    model = UNet1DModel(config)
    lightning_module = DenoisersLightningModule(model)

    audio = torch.randn(1, 1, config.max_length)
    batch = Batch(audio=audio, noisy=audio, lengths=torch.tensor([audio.shape[-1]]))

    # test forward
    with torch.no_grad():
        recon = lightning_module(audio).audio

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape

    # test training step
    loss = lightning_module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # test validation step
    loss = lightning_module.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

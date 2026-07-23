"""Tests for FlowUNet1D model."""

import torch

from denoisers.modeling.flowunet1d.model import FlowUNet1DConfig, FlowUNet1DModel


def test_config():
    """Test config."""
    config = FlowUNet1DConfig(
        max_length=8192,
        sample_rate=16000,
        channels=(1, 2, 3, 4, 5, 6),
        kernel_size=3,
        dropout=0.1,
        activation="silu",
        sigma_max=0.05,
        num_inference_steps=2,
        stochastic_sampling=True,
    )
    assert config.max_length == 8192
    assert config.sample_rate == 16000
    assert config.channels == (1, 2, 3, 4, 5, 6)
    assert config.kernel_size == 3
    assert config.dropout == 0.1
    assert config.activation == "silu"
    assert config.sigma_max == 0.05
    assert config.num_inference_steps == 2
    assert config.stochastic_sampling is True


def test_model() -> None:
    """Test sampling forward pass."""
    config = FlowUNet1DConfig(
        max_length=16384,
        sample_rate=16000,
        channels=(2, 4, 6, 8),
        kernel_size=3,
        num_groups=2,
        num_inference_steps=2,
    )
    model = FlowUNet1DModel(config)
    model.eval()

    audio = torch.randn(1, 1, config.max_length)
    with torch.no_grad():
        recon = model(audio).audio

    assert isinstance(recon, torch.Tensor)
    assert audio.shape == recon.shape


def test_single_step_equals_predictive_call() -> None:
    """One sampling step must equal a single clean prediction from t=1."""
    config = FlowUNet1DConfig(
        max_length=16384,
        sample_rate=16000,
        channels=(2, 4, 6, 8),
        kernel_size=3,
        num_groups=2,
    )
    model = FlowUNet1DModel(config)
    model.eval()

    noisy = torch.randn(1, 1, config.max_length)
    with torch.no_grad():
        sampled = model(noisy, num_steps=1, stochastic=False).audio
        predicted = model.predict_clean(
            noisy, noisy, torch.ones(1, dtype=noisy.dtype)
        )

    assert torch.allclose(sampled, predicted)


def test_bridge_sigma_endpoints() -> None:
    """Bridge noise must vanish at both endpoints."""
    config = FlowUNet1DConfig(channels=(2, 4, 6, 8), num_groups=2, sigma_max=0.1)
    model = FlowUNet1DModel(config)

    t = torch.tensor([0.0, 0.5, 1.0])
    sigma = model.bridge_sigma(t)

    assert sigma[0] == 0.0
    assert sigma[2] == 0.0
    assert sigma[1] > 0.0

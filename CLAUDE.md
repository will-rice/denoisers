# CLAUDE.md

## Project Overview

**Denoisers** is a Python library for training and using audio denoising models. It provides two neural network architectures (WaveUNet and UNet1D) built on the Hugging Face `transformers` `PreTrainedModel` API, with PyTorch Lightning for training orchestration.

- **Repository**: https://github.com/will-rice/denoisers
- **Version**: 0.2.0
- **License**: Apache 2.0
- **Python**: >=3.9 (CI tests on 3.11, 3.12, 3.13)
- **Package manager**: `uv`

## Quick Reference Commands

```bash
# Install dependencies
uv sync

# Run all checks (linting, type checking, tests) — same as CI
uv run pre-commit run -a

# Run individual checks
uv run ruff check . --fix    # Lint + auto-fix
uv run mypy .                # Type checking
uv run python -m pytest      # Tests

# Train a model
uv run train <model> <name> <data_root> [options]
# Example: uv run train unet1d my-model ./data --batch_size 32
```

## Project Structure

```
src/denoisers/
├── __init__.py                  # Public API: WaveUNetConfig, WaveUNetModel, UNet1DConfig, UNet1DModel
├── modeling/
│   ├── __init__.py              # MODELS and CONFIGS registries
│   ├── modules.py               # Shared modules (Upsample1D, Downsample1D)
│   ├── waveunet/                # WaveUNet architecture
│   │   ├── config.py            # WaveUNetConfig
│   │   ├── model.py             # WaveUNetModel (PreTrainedModel)
│   │   └── modules.py           # WaveUNet-specific layers
│   └── unet1d/                  # UNet1D architecture
│       ├── config.py            # UNet1DConfig
│       ├── model.py             # UNet1DModel (PreTrainedModel)
│       └── modules.py           # UNet1D-specific layers
├── datasets/
│   └── audio.py                 # AudioDataset with on-the-fly noise augmentation
├── datamodule.py                # PyTorch Lightning DataModule (95/5 train/val split)
├── lightning_module.py          # DenoisersLightningModule (training logic)
├── losses.py                    # MultiResolutionSTFTLoss, SpectralConvergenceLoss, LogSTFTMagnitudeLoss
├── metrics.py                   # PESQ, DNSMOS evaluation metrics
├── transforms.py                # Audio transforms
├── utils.py                     # Logging and plotting utilities
├── testing/
│   └── __init__.py              # sine_wave() test helper
└── scripts/
    ├── train.py                 # CLI training entry point
    └── publish.py               # Publish models to Hugging Face Hub
```

```
tests/
├── modeling/
│   ├── test_waveunet_model.py   # WaveUNet forward pass tests
│   ├── test_unet1d_model.py     # UNet1D forward pass tests
│   ├── test_lightning_module.py # Training step tests
│   ├── test_pretrained.py       # Pretrained model loading tests
│   └── test_modules.py         # Shared module tests
├── datasets/
│   └── test_audio.py            # Dataset tests
├── test_metrics.py              # Metrics tests
├── test_transforms.py           # Transform tests
└── assets/reverb/               # Test audio files
```

## Architecture

Both models subclass `transformers.PreTrainedModel` and follow the Hugging Face pattern:
- Each has a `Config` (subclass of `PretrainedConfig`) and a `Model`
- Models output a named tuple with `.audio` (denoised) and `.noise` (estimated noise)
- Both support `autoencoder` mode (predict clean audio) and noise prediction mode
- Models are registered in `MODELS` and `CONFIGS` dicts in `denoisers.modeling.__init__`

**WaveUNet**: Encoder-decoder with skip connections. 12-layer progressive channel expansion (24→288). Kernel sizes: down=15, up=5.

**UNet1D**: Modern architecture inspired by diffusion models. 12-layer channel expansion (32→384). Conv1D-based with SiLU activation.

## Code Style and Conventions

### Linting (Ruff)
- Rules: C, E, F, I, W, D, N, B
- Docstring convention: **Google style** (`D` rules with `convention = "google"`)
- `D107` is ignored (missing docstring in `__init__`)
- Import sorting: `isort` with `denoisers` as first-party

### Type Checking (MyPy)
- `ignore_missing_imports = true`
- `strict = false`

### Docstrings
All public modules, classes, and functions require Google-style docstrings with Args, Returns, and type annotations in the docstring.

### Testing
- Framework: `pytest`
- DeprecationWarnings are filtered/ignored
- Use `denoisers.testing.sine_wave()` to generate test audio tensors
- Tests mirror the source structure under `tests/`

## Pre-commit Hooks

Pre-commit runs these hooks in order (all must pass):
1. `check-ast` — Validate Python syntax
2. `end-of-file-fixer` — Ensure files end with newline
3. `trailing-whitespace` — Remove trailing whitespace
4. `check-merge-conflict` — No merge conflict markers
5. `requirements-txt-fixer` — Sort requirements.txt
6. `mdformat` — Format markdown (GFM + black style)
7. `ruff` — Lint and auto-fix (`uv run ruff check . --fix`)
8. `mypy` — Type checking (`uv run mypy .`)
9. `pytest` — Run test suite (`uv run python -m pytest`)

## CI/CD

- **CI** (`.github/workflows/python-package.yml`): Runs `uv run pre-commit run -a` on push/PR to `main` across Python 3.11, 3.12, 3.13
- **Publish** (`.github/workflows/python-publish.yml`): Builds and publishes to PyPI on GitHub release

## Key Patterns

- All commands use `uv run` — do not use `pip` or bare `python` directly
- Models follow the Hugging Face `PreTrainedModel` / `PretrainedConfig` pattern for save/load/push_to_hub
- Training uses PyTorch Lightning with W&B logging, DeepSpeed Stage 2 for multi-GPU, and bf16 mixed precision
- Loss: L1 + MultiResolutionSTFTLoss (spectral convergence + log STFT magnitude)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2) with ExponentialLR scheduler (gamma=0.999875)
- Audio data augmentation: room impulse response simulation, colored noise, Gaussian noise via `audiomentations` and `pyroomacoustics`

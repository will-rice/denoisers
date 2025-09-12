# Denoisers

[![PyPI version](https://badge.fury.io/py/denoisers.svg)](https://badge.fury.io/py/denoisers)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wrice/denoisers)

Denoisers is a comprehensive deep learning library for audio denoising with a focus on simplicity, flexibility, and state-of-the-art performance. The library provides two main neural network architectures optimized for different use cases: WaveUNet for high-quality waveform processing and UNet1D for efficient real-time applications.

## 🎯 Key Features

- **Two State-of-the-Art Architectures**:

  - **WaveUNet**: Based on the [original paper](https://arxiv.org/abs/1806.03185), optimized for high-fidelity audio restoration
  - **UNet1D**: Custom architecture inspired by diffusion models, designed for efficiency and real-time processing

- **Pre-trained Models**: Ready-to-use models available on Hugging Face Hub

- **Easy Integration**: Simple API for both inference and training

- **Production Ready**: Built with PyTorch Lightning for scalable training and deployment

- **Comprehensive Metrics**: Advanced audio quality metrics including PESQ, STOI, and DNS-MOS

- **Flexible Training**: Support for various loss functions, data augmentation, and training strategies

## 🚀 Quick Start

### Installation

```bash
pip install denoisers
```

### Basic Usage

#### Inference with Pre-trained Models

```python
import torch
import torchaudio
from denoisers import WaveUNetModel
from tqdm import tqdm

# Load pre-trained model
model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-24khz")

# Load and preprocess audio
audio, sr = torchaudio.load("noisy_audio.wav")

# Resample if necessary
if sr != model.config.sample_rate:
    audio = torchaudio.functional.resample(audio, sr, model.config.sample_rate)

# Convert to mono if stereo
if audio.size(0) > 1:
    audio = audio.mean(0, keepdim=True)

# Process audio in chunks to handle long files
chunk_size = model.config.max_length
padding = abs(audio.size(-1) % chunk_size - chunk_size)
padded = torch.nn.functional.pad(audio, (0, padding))

clean = []
for i in tqdm(range(0, padded.shape[-1], chunk_size)):
    audio_chunk = padded[:, i : i + chunk_size]
    with torch.no_grad():
        clean_chunk = model(audio_chunk[None]).audio
    clean.append(clean_chunk.squeeze(0))

# Concatenate results and remove padding
denoised = torch.concat(clean, 1)[:, : audio.shape[-1]]

# Save denoised audio
torchaudio.save("clean_audio.wav", denoised, model.config.sample_rate)
```

#### Available Pre-trained Models

| Model                        | Sample Rate | Architecture | Use Case                        |
| ---------------------------- | ----------- | ------------ | ------------------------------- |
| `wrice/waveunet-vctk-24khz`  | 24kHz       | WaveUNet     | High-quality speech denoising   |
| `wrice/unet1d-general-48khz` | 48kHz       | UNet1D       | General-purpose audio denoising |

## 🏗️ Architecture Overview

### WaveUNet

The WaveUNet architecture implements a U-Net style network specifically designed for waveform processing:

- **Encoder-Decoder Architecture**: Progressive downsampling followed by upsampling with skip connections
- **Waveform Processing**: Direct operation on raw audio waveforms
- **High Quality**: Optimized for maximum audio fidelity
- **Configurable Depth**: Adjustable network depth for different complexity requirements

**Key Parameters:**

- `in_channels`: Channel progression through the network (default: 24→288)
- `downsample_kernel_size`: Kernel size for downsampling layers (default: 15)
- `upsample_kernel_size`: Kernel size for upsampling layers (default: 5)
- `activation`: Activation function (default: "leaky_relu")
- `max_length`: Maximum input length (default: 163,840 samples)

### UNet1D

The UNet1D architecture is a custom implementation inspired by modern diffusion models:

- **Efficient Design**: Optimized for computational efficiency and memory usage
- **Real-time Capable**: Suitable for real-time audio processing applications
- **Modern Architecture**: Incorporates latest advances in deep learning for audio
- **Flexible Configuration**: Highly configurable for different audio types and quality requirements

**Key Parameters:**

- `channels`: Channel progression (default: 32→384)
- `kernel_size`: Convolution kernel size (default: 3)
- `activation`: Activation function (default: "silu")
- `max_length`: Maximum input length (default: 48,000 samples)

## 🔧 Training Your Own Models

### Data Preparation

Organize your training data in the following structure:

```
data_root/
├── clean_audio1.wav
├── clean_audio2.flac
├── clean_audio3.mp3
└── ...
```

Supported formats: `.wav`, `.flac`, `.mp3`, `.ogg`

### Training Command

```bash
# Train a WaveUNet model
train waveunet my-model-name /path/to/data_root/ \
    --batch_size 32 \
    --num_devices 2 \
    --ema \
    --push_to_hub

# Train a UNet1D model
train unet1d my-unet1d-model /path/to/data_root/ \
    --batch_size 64 \
    --num_workers 8 \
    --seed 42
```

### Training Parameters

| Parameter       | Description                                   | Default               |
| --------------- | --------------------------------------------- | --------------------- |
| `--batch_size`  | Training batch size                           | 64                    |
| `--num_devices` | Number of GPUs to use                         | 1 (if CUDA available) |
| `--num_workers` | Data loading workers                          | 4                     |
| `--ema`         | Enable Exponential Moving Average             | False                 |
| `--push_to_hub` | Push model to Hugging Face Hub after training | False                 |
| `--seed`        | Random seed for reproducibility               | 1234                  |
| `--project`     | Weights & Biases project name                 | "denoisers"           |

### Custom Configuration

Create custom model configurations by extending the base config classes:

```python
from denoisers.modeling.waveunet.config import WaveUNetConfig

# Custom WaveUNet configuration
config = WaveUNetConfig(
    in_channels=(32, 64, 128, 256, 512),
    sample_rate=16000,
    max_length=32768,
    activation="relu",
    dropout=0.2,
)
```

## 📊 Loss Functions and Metrics

### Loss Functions

The library implements several advanced loss functions optimized for audio denoising:

- **L1 Loss**: Basic reconstruction loss
- **Multi-Resolution STFT Loss**: Frequency-domain loss for better perceptual quality
  - Spectral Convergence Loss
  - Log STFT Magnitude Loss
  - Multiple resolution scales for comprehensive frequency coverage

### Evaluation Metrics

Comprehensive audio quality assessment:

- **SNR**: Signal-to-Noise Ratio
- **SDR**: Signal-to-Distortion Ratio
- **SI-SNR**: Scale-Invariant Signal-to-Noise Ratio
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **DNS-MOS**: Deep Noise Suppression Mean Opinion Score
- **PESQ**: Perceptual Evaluation of Speech Quality

## 🔬 Advanced Features

### Data Augmentation

Built-in audio augmentation pipeline using `audiomentations`:

- **Gaussian Noise Addition**: Simulates various noise conditions
- **Colored Noise**: Pink, brown, and other colored noise types
- **Room Simulation**: Realistic acoustic environments using `pyroomacoustics`
- **Dynamic Sample Rate**: Training with multiple sample rates for robustness

### Exponential Moving Average (EMA)

Optional EMA model averaging for improved training stability and performance:

```bash
train waveunet my-model /data/ --ema
```

### Mixed Precision Training

Automatic mixed precision training with PyTorch Lightning for faster training and reduced memory usage.

### Weights & Biases Integration

Built-in experiment tracking and visualization:

- Training and validation metrics
- Audio samples logging
- Model checkpoints
- Hyperparameter tracking

## 📈 Model Publishing

Easily share your trained models on Hugging Face Hub:

```bash
publish waveunet my-awesome-model /path/to/model/checkpoint
```

This automatically:

- Uploads model weights and configuration
- Creates model cards with training details
- Enables easy model sharing and distribution

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/will-rice/denoisers.git
cd denoisers

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project maintains high code quality standards:

- **Type Checking**: MyPy static type analysis
- **Linting**: Ruff for code formatting and style
- **Testing**: Pytest with comprehensive test coverage
- **Documentation**: Google-style docstrings

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=denoisers --cov-report=html
```

## 📚 API Reference

### Core Models

#### WaveUNetModel

```python
class WaveUNetModel(PreTrainedModel):
    """WaveUNet model for audio denoising."""

    def forward(self, inputs: torch.Tensor) -> WaveUNetModelOutputs:
        """
        Args:
            inputs: Noisy audio tensor [batch_size, channels, length]

        Returns:
            WaveUNetModelOutputs with .audio attribute containing cleaned audio
        """
```

#### UNet1DModel

```python
class UNet1DModel(PreTrainedModel):
    """UNet1D model for efficient audio denoising."""

    def forward(self, inputs: torch.Tensor) -> UNet1DModelOutputs:
        """
        Args:
            inputs: Noisy audio tensor [batch_size, channels, length]

        Returns:
            UNet1DModelOutputs with .audio attribute containing cleaned audio
        """
```

### Datasets

#### AudioDataset

```python
class AudioDataset(Dataset):
    """Audio dataset with automatic noise synthesis."""

    def __init__(
        self,
        root: Path,
        max_length: int,
        sample_rate: int,
        variable_sample_rate: bool = True,
    ):
        """
        Args:
            root: Path to audio files
            max_length: Maximum audio length in samples
            sample_rate: Target sample rate
            variable_sample_rate: Whether to use variable sample rates during training
        """
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and standards
- Testing requirements
- Pull request process

### Areas for Contribution

- New model architectures
- Additional loss functions
- Enhanced data augmentation
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WaveUNet paper](https://arxiv.org/abs/1806.03185) for the original architecture
- PyTorch Lightning team for the excellent training framework
- Hugging Face for model hosting and distribution
- The open-source audio processing community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/will-rice/denoisers/issues)
- **Discussions**: [GitHub Discussions](https://github.com/will-rice/denoisers/discussions)
- **Email**: wrice20@gmail.com

## 🔗 Links

- [Hugging Face Demo](https://huggingface.co/spaces/wrice/denoisers)
- [PyPI Package](https://pypi.org/project/denoisers/)
- [Documentation](https://github.com/will-rice/denoisers/wiki)
- [Model Zoo](https://huggingface.co/wrice)

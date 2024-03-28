# Denoisers

Denoisers is a denoising library for audio with a focus on simplicity and ease of use. There are two major architectures available for waveforms: WaveUNet which follows the [paper](https://arxiv.org/abs/1806.03185) and a custom UNet1D architecture similar to what you would see in diffusion models.

## Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wrice/denoisers)


## Usage/Examples

```sh
pip install denoisers
```

```python
import torch
import torchaudio
from denoisers import WaveUNetModel
from tqdm import tqdm

model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-24khz")

audio, sr = torchaudio.load("noisy_audio.wav")

if sr != model.config.sample_rate:
    audio = torchaudio.functional.resample(audio, sr, model.config.sample_rate)

if audio.size(0) > 1:
    audio = audio.mean(0, keepdim=True)

chunk_size = model.config.max_length

padding = abs(audio.size(-1) % chunk_size - chunk_size)
padded = torch.nn.functional.pad(audio, (0, padding))

clean = []
for i in tqdm(range(0, padded.shape[-1], chunk_size)):
    audio_chunk = padded[:, i:i + chunk_size]
    with torch.no_grad():
        clean_chunk = model(audio_chunk[None]).audio
    clean.append(clean_chunk.squeeze(0))

denoised = torch.concat(clean, 1)[:, :audio.shape[-1]]
```

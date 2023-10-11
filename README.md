# Denoisers

Denoisers is a denoising library for audio with a focus on simplicity and ease of use. There are two major types of architectures available. WaveUNet for waveform denoising and UNet for spectrogram denoising.

## Usage/Examples

```python
import torch
import torchaudio
from denoisers import WaveUNetModel
from tqdm import tqdm

model = WaveUNetModel.from_pretrained("wrice/waveunet-vctk-24khz")

audio, sr = torchaudio.load("noisy_audio.wav")

if sr != model.config.sample_rate:
    audio = torchaudio.functional.resample(audio, sr, model.config.sample_rate)

chunk_size = model.config.max_length

padding = abs(audio.size(-1) % chunk_size - chunk_size)
padded = torch.nn.functional.pad(audio, (0, padding))

clean = []
for i in tqdm(range(0, padded.shape[-1], chunk_size)):
    audio_chunk = padded[:, i:i+chunk_size].cuda()
    with torch.no_grad():
        clean_chunk = model(audio_chunk[None])
    clean.append(clean_chunk.squeeze(0).cpu())

denoised = torch.concat(clean)[:, :audio.shape[-1]]
```

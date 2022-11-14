"""VCTK dataset."""
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch import Tensor, nn
from torch.utils.data import Dataset

from src.denoiser.transforms import (
    BreakTransform,
    ClipTransform,
    CutOut,
    FilterTransform,
    FreqNoiseMask,
    GaussianNoise,
    NoiseFromFile,
    NoiseOut,
    ReverbFromSoundboard,
    SpecTransform,
    TimeNoiseMask,
    VolTransform,
)


class Sample(NamedTuple):
    """Sample object for easy access to model inputs."""

    audio: Tensor
    noisy_audio: Tensor
    audio_lengths: Tensor
    specs: Tensor
    noisy_specs: Tensor
    spec_lengths: Tensor


class VCTKDataset(Dataset):
    """Simple dataset."""

    def __init__(
        self,
        root: Path,
        max_length: int,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
        sample_rate: int = 24000,
        transforms=None,
    ):
        super().__init__()
        self._root = root
        self._max_length = max_length
        self._n_fft = n_fft
        self._win_length = win_length
        self._hop_length = hop_length
        self._sample_rate = sample_rate

        self._transforms = transforms or nn.Sequential(
            ReverbFromSoundboard(p=0.99),
            GaussianNoise(p=0.9),
            VolTransform(),
            FilterTransform(),
            # ClipTransform(),
            # BreakTransform(),
            SpecTransform(),
            FreqNoiseMask(100, p=0.5),
            TimeNoiseMask(100, p=0.5),
            NoiseOut(20, 5),
            CutOut(20, 5),
            NoiseFromFile(Path("/data/daps")),
        )

        self._samples = list(self._root.glob("**/*.flac"))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._samples[idx]

        audio, sr = torchaudio.load(sample)

        if sr != self._sample_rate:
            audio = AF.resample(audio, sr, self._sample_rate)

        audio = audio[0].squeeze(0)
        audio_length = audio.size(0)

        noisy = torch.clone(audio)
        noisy = self._transforms(noisy)
        noisy = torch.FloatTensor(noisy)

        if audio_length < self._max_length:
            pad_length = self._max_length - audio_length
            padded = F.pad(audio, (0, pad_length))
            noisy = F.pad(noisy, (0, pad_length))
        else:
            padded = audio[: self._max_length]
            noisy = noisy[: self._max_length]

        spec = self.get_spectrogram(padded)
        noisy_spec = self.get_spectrogram(noisy)
        spec_length = spec.size(1)

        return Sample(
            audio=padded.unsqueeze(0),
            noisy_audio=noisy.unsqueeze(0),
            audio_lengths=audio_length,
            specs=spec,
            noisy_specs=noisy_spec,
            spec_lengths=spec_length,
        )

    def get_spectrogram(self, inputs: Tensor) -> Tensor:
        """Calculate magnitude spectrogram."""
        spec = torch.stft(
            inputs,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            return_complex=True,
        ).abs()

        return spec

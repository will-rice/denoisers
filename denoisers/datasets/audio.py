"""Audio dataset."""
import random
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio
from audiomentations import AddColorNoise, AddGaussianNoise, Compose, RoomSimulator
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3"}


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    noisy: torch.Tensor
    lengths: torch.Tensor


class AudioDataset(Dataset):
    """Simple audio dataset."""

    def __init__(self, root: Path, max_length: int, sample_rate: int) -> None:
        super().__init__()
        self._root = root
        self._samples = []
        for ext in SUPPORTED_EXTENSIONS:
            self._samples.extend(list(self._root.glob(f"**/*{ext}")))
        self._max_length = max_length
        self._sample_rate = sample_rate

        self._transforms = Compose(
            [
                RoomSimulator(p=0.8, leave_length_unchanged=True),
                AddColorNoise(p=0.97),
                AddGaussianNoise(p=0.97),
            ]
        )

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Batch:
        """Return item from dataset."""
        path = self._samples[idx]
        audio, sr = torchaudio.load(str(path))

        if sr != self._sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self._sample_rate)

        if audio.size(0) > 1:
            audio = audio.mean(0, keepdim=True)

        audio_length = min(audio.size(-1), self._max_length)

        if audio_length < self._max_length:
            pad_length = self._max_length - audio_length
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        else:
            start_idx = random.randint(0, audio.size(-1) - self._max_length)
            audio = audio[:, start_idx : start_idx + self._max_length]

        noisy = self._transforms(audio.clone().numpy(), sample_rate=self._sample_rate)
        noisy = torch.from_numpy(noisy)

        return Batch(audio=audio, noisy=noisy, lengths=torch.tensor(audio_length))

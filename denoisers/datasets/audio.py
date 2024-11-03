"""Audio dataset."""
import random
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from torch_audiomentations import (
    AddColoredNoise,
    ApplyImpulseResponse,
    Compose,
    Identity,
)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3"}


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    noisy: torch.Tensor
    lengths: torch.Tensor


class AudioDataset(Dataset):
    """Simple audio dataset."""

    def __init__(
        self,
        root: Path,
        max_length: int,
        sample_rate: int,
        rir_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._samples = []
        for ext in SUPPORTED_EXTENSIONS:
            self._samples.extend(list(self._root.glob(f"**/*{ext}")))
        self._max_length = max_length
        self._sample_rate = sample_rate
        self._transforms = Compose(
            [
                ApplyImpulseResponse(rir_root, p=0.8) if rir_root else Identity(),
                AddColoredNoise(p=0.97),
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

        noisy = self._transforms(audio[None].clone(), sample_rate=self._sample_rate)[0]

        return Batch(audio=audio, noisy=noisy, lengths=torch.tensor(audio_length))

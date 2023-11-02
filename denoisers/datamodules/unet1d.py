"""Unet1D Data modules."""
import os
from typing import List, NamedTuple, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from denoisers import transforms


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: Tensor
    noisy: Tensor
    lengths: Tensor


class AudioFromFileDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 24,
        num_workers: int = os.cpu_count() // 2,  # type: ignore
        max_length: int = 16384 * 10,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        # we don't use sample_rate here for divisibility
        self._max_length = max_length
        self._sample_rate = sample_rate
        self._transforms = nn.Sequential(
            transforms.ReverbFromSoundboard(p=1.0, sample_rate=sample_rate),
            transforms.GaussianNoise(p=1.0),
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        """Setup datasets."""
        train_split = int(np.floor(len(self._dataset) * 0.95))  # type: ignore
        val_split = int(np.ceil(len(self._dataset) * 0.05))  # type: ignore

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self._dataset, lengths=(train_split, val_split)
        )

    def pad_collate_fn(self, paths: List[str]) -> Batch:
        """Pad collate function."""
        audios = []
        noisy_audio = []
        lengths = []
        for path in paths:
            try:
                audio, sr = torchaudio.load(path)
            except Exception:
                continue

            if sr != self._sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self._sample_rate)

            if audio.size(0) > 1:
                audio = audio[0].unsqueeze(0)

            audio_length = min(audio.size(-1), self._max_length)

            if audio_length < self._max_length:
                pad_length = self._max_length - audio_length
                audio = F.pad(audio, (0, pad_length))
            else:
                audio = audio[:, : self._max_length]

            noisy = self._transforms(audio.clone())

            audios.append(audio)
            noisy_audio.append(noisy)
            lengths.append(torch.tensor(audio_length))

        return Batch(
            audio=torch.stack(audios),
            noisy=torch.stack(noisy_audio),
            lengths=torch.stack(lengths),
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self.pad_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self.pad_collate_fn,
            shuffle=False,
            drop_last=True,
        )

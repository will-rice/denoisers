"""Data modules."""
import os
from pathlib import Path
from typing import Any, NamedTuple, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.denoiser import transforms

MAX_LENGTH = 16384 * 14


class Sample(NamedTuple):
    """Sample object for easy access to model inputs."""

    audio: Tensor
    noisy_audio: Tensor
    audio_lengths: Tensor
    specs: Tensor
    noisy_specs: Tensor
    spec_lengths: Tensor


class HDF5Dataset(Dataset):
    """Dataset for audio files."""

    def __init__(
        self,
        root: Path,
        max_length: int = MAX_LENGTH,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self._root = root
        self.max_length = max_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        samples = h5py.File(root, "r")
        self.samples = list(samples.values())

        self.transforms = transforms.RandomTransform(
            transforms=(
                transforms.ReverbFromSoundboard(p=0.99),
                transforms.GaussianNoise(p=0.9),
                transforms.VolTransform(),
                transforms.FilterTransform(),
                transforms.ClipTransform(),
                transforms.BreakTransform(),
                transforms.SpecTransform(),
                transforms.FreqNoiseMask(100, p=0.5),
                transforms.TimeNoiseMask(100, p=0.5),
            )
        )

    def __getitem__(self, item):
        sample = self.samples[item]
        audio = sample["audio"][:]
        audio = audio.astype(np.float32) / (2**15 - 1)
        audio = torch.FloatTensor(audio)

        audio_length = audio.size(0)

        noisy = torch.clone(audio)
        noisy = self.transforms(noisy)
        noisy = torch.FloatTensor(noisy)

        if audio_length < self.max_length:
            pad_length = self.max_length - audio_length
            padded = F.pad(sample, (0, pad_length))
            noisy = F.pad(noisy, (0, pad_length))
        else:
            padded = sample[: self.max_length]
            noisy = noisy[: self.max_length]

        spec = self.get_spectrogram(padded)
        noisy_spec = self.get_spectrogram(noisy)
        spec_length = spec.size(1)

        return Sample(
            audio=audio,
            noisy_audio=noisy,
            audio_lengths=audio_length,
            specs=spec,
            noisy_specs=noisy_spec,
            spec_lengths=spec_length,
        )

    def __len__(self):
        return len(self.samples)

    def get_spectrogram(self, inputs: Tensor) -> Tensor:
        """Calculate magnitude spectrogram."""
        spec = torch.stft(
            inputs,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            return_complex=True,
        ).abs()

        return spec


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 24,
        num_workers: int = os.cpu_count(),
        max_length: int = MAX_LENGTH,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.transform = transforms.RandomTransform(
            transforms=(
                transforms.ReverbFromSoundboard(p=0.99),
                transforms.GaussianNoise(p=0.9),
                transforms.VolTransform(),
                transforms.FilterTransform(),
                transforms.ClipTransform(),
                transforms.BreakTransform(),
                transforms.SpecTransform(),
                transforms.FreqNoiseMask(100, p=0.5),
                transforms.TimeNoiseMask(100, p=0.5),
            )
        )

    def prepare_data(self) -> None:
        """Download datasets."""
        torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="train-clean-360",
            folder_in_archive="LibriTTS",
            download=True,
        )
        torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="test-clean",
            folder_in_archive="LibriTTS",
            download=True,
        )
        torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="dev-clean",
            folder_in_archive="LibriTTS",
            download=True,
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        """Setup datasets."""
        self.train_dataset = torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="train-clean-360",
            folder_in_archive="LibriTTS",
            download=False,
        )
        self.test_dataset = torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="test-clean",
            folder_in_archive="LibriTTS",
            download=False,
        )
        self.val_dataset = torchaudio.datasets.LIBRITTS(
            root=self.data_dir,
            url="dev-clean",
            folder_in_archive="LibriTTS",
            download=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def pad_collate(self, batch: Any) -> Any:
        """Custom collate function."""
        samples, *_ = zip(*batch)

        audio = []
        audio_lengths = []
        noisy_audio = []
        specs = []
        spec_lengths = []
        noisy_specs = []

        for sample in samples:
            sample = sample.squeeze()
            sample = torch.clamp(sample, -1.0, 1.0)
            audio_length = sample.size(0)

            noisy = torch.clone(sample)
            noisy = self.transform(noisy)
            noisy = torch.FloatTensor(noisy)

            if audio_length < self.max_length:
                pad_length = self.max_length - audio_length
                padded = F.pad(sample, (0, pad_length))
                noisy = F.pad(noisy, (0, pad_length))
            else:
                padded = sample[: self.max_length]
                noisy = noisy[: self.max_length]

            spec = self.get_spectrogram(padded)
            noisy_spec = self.get_spectrogram(noisy)
            spec_length = spec.size(1)

            audio.append(padded)
            audio_lengths.append(torch.tensor(audio_length))
            noisy_audio.append(noisy)
            specs.append(spec)
            spec_lengths.append(torch.tensor(spec_length))
            noisy_specs.append(noisy_spec)

        return Sample(
            audio=torch.stack(audio).unsqueeze(1),
            audio_lengths=torch.stack(audio_lengths),
            noisy_audio=torch.stack(noisy_audio).unsqueeze(1),
            specs=torch.stack(specs).unsqueeze(1),
            spec_lengths=torch.stack(spec_lengths),
            noisy_specs=torch.stack(noisy_specs).unsqueeze(1),
        )

    def get_spectrogram(self, inputs: Tensor) -> Tensor:
        """Calculate magnitude spectrogram."""
        spec = torch.stft(
            inputs,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            return_complex=True,
        ).abs()

        return spec

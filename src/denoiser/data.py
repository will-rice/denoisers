"""Data modules."""
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.denoiser import transforms
from src.denoiser.datasets import Batch
from src.denoiser.datasets.vctk import VCTKDataset

MAX_LENGTH = 16384 * 10


class AudioFromFileDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 24,
        num_workers: int = os.cpu_count() or 12,
        max_length: int = MAX_LENGTH,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def setup(self, stage: Optional[str] = "fit") -> None:
        """Setup datasets."""
        dataset = VCTKDataset(
            self.data_dir,
            max_length=self.max_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )
        train_split = int(np.floor(len(dataset) * 0.95))
        val_split = int(np.ceil(len(dataset) * 0.05))

        assert (train_split + val_split) == len(dataset)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, lengths=(train_split, val_split)
        )

        val_split = len(self.val_dataset) // 2
        test_split = len(self.val_dataset) - val_split
        self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.val_dataset, lengths=(val_split, test_split)
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 24,
        num_workers: int = os.cpu_count() or 12,
        max_length: int = MAX_LENGTH,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.max_length = max_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.transform = transforms.RandomTransform(
            transforms=(
                transforms.ReverbFromSoundboard(p=1.0),
                transforms.GaussianNoise(p=1.0),
                transforms.VolTransform(),
                transforms.FilterTransform(),
                transforms.ClipTransform(),
                # transforms.BreakTransform(),
                transforms.SpecTransform(),
                transforms.FreqNoiseMask(100, p=0.5),
                transforms.TimeNoiseMask(100, p=0.5),
                transforms.NoiseOut(20, 5),
                transforms.CutOut(20, 5),
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

        return Batch(
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

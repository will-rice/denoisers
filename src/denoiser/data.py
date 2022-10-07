"""Data modules."""
import random
from dataclasses import dataclass
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torchaudio
from pedalboard import Reverb
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.denoiser.transforms import GaussianNoise


@dataclass
class Sample:
    """Sample object for easy access to model inputs."""

    audio: Tensor
    noisy_audio: Tensor
    audio_lengths: List[int]
    specs: Tensor
    noisy_specs: Tensor
    spec_lengths: List[int]


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 24,
        max_length: int = 16384,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.noiser = nn.Sequential(GaussianNoise())
        self.reverb = Reverb()

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
            num_workers=24,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=24,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=24,
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
            audio_length = sample.size(0)

            if audio_length < self.max_length:
                padded = F.pad(sample, (0, self.max_length))
            else:
                padded = sample

            random_idx = torch.randint(high=padded.size(0) - self.max_length, size=())
            padded = padded[random_idx : random_idx + self.max_length]

            self.reverb.room_size = random.random()
            noisy = self.reverb.process(padded.detach().numpy(), 24000)
            noisy = torch.FloatTensor(noisy)
            noisy += self.noiser(padded)

            spec = self.get_spectrogram(padded)
            noisy_spec = self.get_spectrogram(noisy)
            spec_length = spec.size(1)

            audio.append(padded)
            audio_lengths.append(audio_length)
            noisy_audio.append(noisy)
            specs.append(spec)
            spec_lengths.append(spec_length)
            noisy_specs.append(noisy_spec)

        return Sample(
            audio=torch.stack(audio).unsqueeze(1),
            audio_lengths=audio_lengths,
            noisy_audio=torch.stack(noisy_audio).unsqueeze(1),
            specs=torch.stack(specs).unsqueeze(1),
            spec_lengths=spec_lengths,
            noisy_specs=torch.stack(noisy_specs).unsqueeze(1),
        )

    def get_spectrogram(self, inputs: Tensor) -> Tensor:
        spec = torch.stft(
            inputs,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            return_complex=True,
        ).abs()

        return spec

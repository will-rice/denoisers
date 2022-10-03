"""Data modules."""
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

MAX_LENGTH = 24000 * 1


class GaussianNoise(nn.Module):
    """Gaussian Noise Transform."""

    def __init__(self, min_intensity: float = 0.0, max_intensity: float = 50.0):
        super().__init__()
        self.intensity_dist = torch.distributions.uniform.Uniform(
            min_intensity, max_intensity
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        intensity = self.intensity_dist.sample().to(x.device)
        noise = torch.randn_like(x) * intensity
        noisy = x + noise
        return noisy


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 24,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noiser = GaussianNoise()
        self.transform = nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power"),
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
        audio, *_ = zip(*batch)

        slices = []
        noise = []
        noisy = []
        for a in audio:
            a = a.squeeze()
            a = F.pad(a, (0, MAX_LENGTH))
            a = a[:MAX_LENGTH]
            s = self.transform(a)
            n = self.noiser(s)
            slices.append(s)
            noisy.append(n)
            noise.append(s - n)

        samples = torch.stack(slices).unsqueeze(1)
        noisys = torch.stack(noisy).unsqueeze(1)
        noises = torch.stack(noise).unsqueeze(1)

        return samples, noisys, noises

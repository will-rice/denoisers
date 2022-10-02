"""Data modules."""
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

MAX_LENGTH = 24000 * 4


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(self, data_dir: str = "./", batch_size: int = 32) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

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
            num_workers=8,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=8,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_collate,
            num_workers=8,
            shuffle=False,
        )

    @staticmethod
    def pad_collate(batch: Any) -> Tensor:
        """Custom collate function."""
        audio, *_ = zip(*batch)

        random_slices = []
        for a in audio:
            a = a.squeeze()
            a = F.pad(a, (0, MAX_LENGTH))
            a = a[:MAX_LENGTH]
            random_slices.append(a)

        samples = torch.stack(random_slices)

        return samples

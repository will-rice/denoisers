"""Data modules."""
from typing import Any, List, NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Sample(NamedTuple):
    """Sample"""

    audio: Tensor
    lengths: List[int]


class LibriTTSDataModule(pl.LightningDataModule):
    """LibriTTS DataModule."""

    def __init__(
        self, data_dir: str = "./", batch_size: int = 24, max_length: int = 24000 * 3
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length

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

        samples = []
        lengths = []
        for a in audio:
            a = a.squeeze()
            audio_length = len(a)
            padded = F.pad(a, (0, self.max_length))
            padded = padded[: self.max_length]
            samples.append(padded)
            lengths.append(audio_length)

        stacked = torch.stack(samples)
        return Sample(audio=stacked, lengths=lengths)

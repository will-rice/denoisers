"""Lightning datamodule."""

from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DenoisersDataModule(LightningDataModule):
    """Lightning DataModule for denoisers."""

    def __init__(
        self, dataset: Dataset, batch_size: int = 24, num_workers: int = 8
    ) -> None:
        super().__init__()

        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage: Optional[str] = "fit") -> None:
        """Split datasets."""
        train_split = int(np.floor(len(self._dataset) * 0.95))  # type: ignore
        val_split = int(np.ceil(len(self._dataset) * 0.05))  # type: ignore

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self._dataset,
            lengths=(train_split, val_split),
        )

    def train_dataloader(self) -> DataLoader:
        """Initialize train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=True,
        )

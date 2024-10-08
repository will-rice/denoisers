"""Audio dataset."""
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class XenoCantoDataset(Dataset):
    """Simple dataset."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root
        self._samples = list(self._root.glob("**/*.flac"))
        self._metadata = pd.read_csv(root / "metadata.csv")
        self._metadata = self._metadata[self._metadata["q"] >= "A"]

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self._metadata)

    def __getitem__(self, idx: int) -> str:
        """Return item from dataset."""
        sample = self._metadata.iloc[idx]
        path = str(self._root / sample.sp / f"XC{sample.id}.flac")
        return path

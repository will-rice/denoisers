"""LibriTTSDataset class."""
from pathlib import Path

from torch.utils.data import Dataset


class LibriTTSDataset(Dataset):
    def __init__(self, root: Path):
        super().__init__()
        self._root = root

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

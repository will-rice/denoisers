"""Audio dataset."""
from pathlib import Path

from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Simple dataset."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root
        self._samples = list(self._root.glob("**/*.flac"))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> str:
        """Return item from dataset."""
        return str(self._samples[idx])

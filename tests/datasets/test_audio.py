"""Test audio datasets."""

from pathlib import Path

import torchaudio

from denoisers.datasets.audio import AudioDataset
from denoisers.testing import sine_wave


def test_audio_dataset(tmpdir):
    """Test audio dataset."""
    save_root = Path(tmpdir) / "test_dataset"
    save_root.mkdir(exist_ok=True, parents=True)
    save_path = save_root / "sample.flac"

    audio = sine_wave(800, 1, 16000)
    torchaudio.save(str(save_path), audio, 16000)

    dataset = AudioDataset(save_path.parent, sample_rate=16000, max_length=16384)
    assert len(dataset) == 1
    batch = dataset[0]
    assert batch.audio.shape == (1, 16384)
    assert batch.noisy.shape == (1, 16384)
    assert batch.lengths.shape == ()

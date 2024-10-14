"""Tests for metrics."""

from denoisers.metrics import PESQ
from denoisers.testing import sine_wave


def test_calculate_pesq() -> None:
    """Test calculate_pesq."""
    pesq = PESQ(sample_rate=16000)
    audio = sine_wave(800, 1, 16000)[None]
    pred = sine_wave(800, 1, 16000)[None]
    score = pesq(pred, audio)
    assert score > 0.0

"""Tests for metrics."""
from denoisers.metrics import calculate_pesq
from denoisers.testing import sine_wave


def test_calculate_pesq():
    """Test calculate_pesq."""
    audio = sine_wave(800, 1, 16000)
    pred = sine_wave(800, 1, 16000)
    pesq = calculate_pesq(pred, audio, 16000)
    assert pesq > 0.0

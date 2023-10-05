"""Metrics for denoising."""
import torch
import torchaudio
import torchmetrics
from pesq import cypesq
from torch import Tensor


def calculate_pesq(pred: Tensor, true: Tensor, sample_rate: int = 24000) -> Tensor:
    """Calculate PESQ."""
    pred_resample = torchaudio.functional.resample(pred, sample_rate, 16000)
    true_resample = torchaudio.functional.resample(true, sample_rate, 16000)

    try:
        pesq = torchmetrics.functional.audio.pesq.perceptual_evaluation_speech_quality(
            pred_resample, true_resample, 16000, "wb"
        )
    except cypesq.NoUtterancesError:
        pesq = torch.tensor(0.0)

    return pesq.mean()

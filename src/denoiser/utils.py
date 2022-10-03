"""General utilities for the denoiser."""
from typing import Any, Optional

import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
import wandb

SPEC_FN = T.Spectrogram(
    n_fft=2048,
    win_length=1024,
    hop_length=256,
    center=True,
    pad_mode="constant",
    power=2.0,
)


def sequence_mask(length: Any, max_length: Optional[Any] = None) -> torch.Tensor:
    """Create a boolean mask from sequence lengths."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    mask = x.unsqueeze(0) < length.unsqueeze(1)
    return mask.unsqueeze(1)


def plot_image_batch(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    preds: torch.Tensor,
    name: str,
) -> None:
    """Plot a batch of images and log them to wandb."""
    np_clean = clean.squeeze(1).cpu().detach().numpy()[:5]
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()[:5]
    np_preds = preds.squeeze(1).cpu().detach().numpy()[:5]

    fig, ax = plt.subplots(len(np_clean), 3, figsize=(20, 5 * len(np_clean)))
    for i, (c, n, p) in enumerate(zip(np_clean, np_noisy, np_preds)):
        ax[i][0].imshow(c, origin="lower", aspect="auto")
        ax[i][0].axis("off")
        ax[i][0].title.set_text("clean")

        ax[i][1].imshow(n, origin="lower", aspect="auto")
        ax[i][1].axis("off")
        ax[i][1].title.set_text("noisy")

        ax[i][2].imshow(p, origin="lower", aspect="auto")
        ax[i][2].axis("off")
        ax[i][2].title.set_text("preds")

    wandb.log({f"{name}_images": wandb.Image(fig)})

    plt.close()


def plot_image_from_audio(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    preds: torch.Tensor,
    lengths: torch.Tensor,
    name: str,
) -> None:
    """Plot a batch of images and log them to wandb."""
    clean = clean.squeeze(1).cpu().detach()[:5]
    noisy = noisy.squeeze(1).cpu().detach()[:5]
    preds = preds.squeeze(1).cpu().detach()[:5]

    fig, ax = plt.subplots(len(clean), 3, figsize=(20, 5 * len(clean)))

    for i, (c, n, p, l) in enumerate(zip(clean, noisy, preds, lengths)):

        original_spec = librosa.power_to_db(SPEC_FN(c[:l]))
        noisy_spec = librosa.power_to_db(SPEC_FN(n[:l]))
        pred_spec = librosa.power_to_db(SPEC_FN(p[:l]))

        ax[i][0].imshow(original_spec, origin="lower", aspect="auto")
        ax[i][0].axis("off")
        ax[i][0].title.set_text("clean")

        ax[i][1].imshow(noisy_spec, origin="lower", aspect="auto")
        ax[i][1].axis("off")
        ax[i][1].title.set_text("noisy")

        ax[i][2].imshow(pred_spec, origin="lower", aspect="auto")
        ax[i][2].axis("off")
        ax[i][2].title.set_text("preds")

    wandb.log({f"{name}_images": wandb.Image(fig)})
    plt.close()


def log_audio_batch(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    preds: torch.Tensor,
    lengths: torch.Tensor,
    name: str,
) -> None:
    """Log a batch of audio to wandb."""
    np_clean = clean.squeeze(1).cpu().detach().numpy()[0][: int(lengths[0])]
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()[0][: int(lengths[0])]
    np_preds = preds.squeeze(1).cpu().detach().numpy()[0][: int(lengths[0])]

    wandb.log(
        {
            f"{name}_audio": {
                f"{name}_clean": wandb.Audio(np_clean, sample_rate=24000),
                f"{name}_noisy": wandb.Audio(np_noisy, sample_rate=24000),
                f"{name}_pred": wandb.Audio(np_preds, sample_rate=24000),
            }
        }
    )

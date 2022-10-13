import matplotlib.pyplot as plt
import torch

import wandb


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    mask = x.unsqueeze(0) < length.unsqueeze(1)
    return mask


def plot_image_batch(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    pred: torch.Tensor,
    name: str,
) -> None:
    np_clean = clean.squeeze(1).cpu().detach().numpy()[:5]
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()[:5]
    np_pred = pred.squeeze(1).cpu().detach().numpy()[:5]

    fig, ax = plt.subplots(len(np_clean), 3, figsize=(20, 5 * len(np_clean)))
    for i, (c, n, p) in enumerate(zip(np_clean, np_noisy, np_pred)):
        ax[i][0].imshow(c, origin="lower", aspect="auto")
        ax[i][0].axis("off")

        ax[i][1].imshow(n, origin="lower", aspect="auto")
        ax[i][1].axis("off")

        ax[i][2].imshow(p, origin="lower", aspect="auto")
        ax[i][2].axis("off")

    wandb.log({f"{name}_images": wandb.Image(fig)})

    plt.close()


def log_audio_batch(
    clean: torch.Tensor, noisy: torch.Tensor, pred: torch.Tensor, name: str
) -> None:
    np_clean = clean.squeeze(1).cpu().detach().numpy()[0]
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()[0]
    np_pred = pred.squeeze(1).cpu().detach().numpy()[0]

    wandb.log(
        {
            f"{name}_audio": {
                f"{name}_clean": wandb.Audio(np_clean, sample_rate=24000),
                f"{name}_noisy": wandb.Audio(np_noisy, sample_rate=24000),
                f"{name}_pred": wandb.Audio(np_pred, sample_rate=24000),
            }
        }
    )

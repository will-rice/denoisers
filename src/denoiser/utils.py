import matplotlib.pyplot as plt
import torch

import wandb


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
    np_clean = clean.squeeze(1).cpu().detach().numpy()[:5]
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()[:5]
    np_pred = pred.squeeze(1).cpu().detach().numpy()[:5]

    for i, (c, n, p) in enumerate(zip(np_clean, np_noisy, np_pred)):
        wandb.log(
            {
                f"{name}_clean_{i}": wandb.Audio(c, sample_rate=24000),
                f"{name}_noisy_{i}": wandb.Audio(n, sample_rate=24000),
                f"{name}_pred_{i}": wandb.Audio(p, sample_rate=24000),
            }
        )

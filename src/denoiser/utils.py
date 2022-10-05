import matplotlib.pyplot as plt
import torch


def plot_image_batch(
    clean: torch.Tensor, noisy: torch.Tensor, pred: torch.Tensor
) -> plt.Figure:
    np_clean = clean.squeeze(1).cpu().detach().numpy()
    np_noisy = noisy.squeeze(1).cpu().detach().numpy()
    np_pred = pred.squeeze(1).cpu().detach().numpy()

    fig, ax = plt.subplots(len(clean), 3, figsize=(20, 5 * len(clean)))
    for i, (c, n, p) in enumerate(zip(np_clean, np_noisy, np_pred)):
        ax[i][0].imshow(c, origin="lower", aspect="auto")
        ax[i][1].imshow(n, origin="lower", aspect="auto")
        ax[i][2].imshow(p, origin="lower", aspect="auto")

    plt.axis("off")
    return fig

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy import signal
from tqdm import tqdm
import time
from csng.utils import crop


def do_run(model, energy_fn, energy_scale, num_timesteps, num_samples=1, progressive=True, desc="progress", grayscale=True, init_imgs=None, approximate_xstart_for_energy=True):
    cur_t = num_timesteps - 1
    stim_pred_history = []
    energy_history = []

    samples = model.sample(
        energy_fn=energy_fn,
        energy_scale=energy_scale,
        num_samples=num_samples,
        init_imgs=init_imgs,
        approximate_xstart_for_energy=approximate_xstart_for_energy,
    )

    for j, samples_t in enumerate(samples):
        cur_t -= 1
        # if (j % 10 == 0 and progressive) or cur_t == -1:
        #     energy = energy_fn(samples_t["pred_xstart"])
        #     curr_imgs = []
        #     for k, image in enumerate(samples_t["pred_xstart"]):
        #         breakpoint()
        #         image = image.detach().cpu()
        #         if grayscale:
        #             image = image.mean(0, keepdim=True)
        #         image = image.add(1).div(2)
        #         image = image.clamp(0, 1)

        #         tqdm.write(
        #             f'step {j} | train energy: {energy["train"]:.4g}'
        #         )
        #         curr_imgs.append(image)
        energy_history.append(energy_fn(samples_t["pred_xstart"], t=0)["train"].detach())
        stim_pred_history.append(samples_t["pred_xstart"].mean(dim=1, keepdim=True).detach().cpu().numpy())

    stim_pred = F.interpolate(
        samples_t["pred_xstart"].mean(dim=1, keepdim=True).detach(),
        size=(36, 64),
        mode="bilinear",
        align_corners=False,
    )

    return energy_history, stim_pred, stim_pred_history


def energy_fn(
        x,
        encoder_model,
        target_response=None,
        norm=60,
        em_weight=1,
        dm_weight=1,
        dm_loss_fn=F.mse_loss,
        xs_zero_to_match=None,
        crop_win=(22,36),
        energy_freq=1,
        t=None,
    ):
    assert energy_freq == 1 or t is not None

    ### skip
    if energy_freq > 1 and t % energy_freq != 0:
        return None

    energy = 0

    ### encoder matching
    tar = F.interpolate(
        x.clone(), size=(36, 64), mode="bilinear", align_corners=False
    ).mean(1, keepdim=True)
    tar = tar / torch.norm(tar, dim=(2,3), keepdim=True) * norm
    resp_pred = encoder_model(tar)
    energy += em_weight * torch.mean((resp_pred - target_response) ** 2, dim=1).sum()

    ### decoder matching
    # tar = F.interpolate(
    #     x.clone(), size=crop_win, mode="bilinear", align_corners=False
    # ).mean(1, keepdim=True)
    tar = crop(tar, crop_win)
    tar = tar / torch.norm(tar, dim=(2,3), keepdim=True) * norm
    for x_zero_to_match in xs_zero_to_match:
        energy += dm_weight * dm_loss_fn(tar, x_zero_to_match)

    return {"train": energy}


def plot_diffusion(
        target_image,
        imgs,
        timesteps=(0, 10, 100, 200, 300, 400, 600, 800, 999),
        crop_win=None,
        save_to=None,
        show=True
    ):
    ### plot progression in one plot
    fig = plt.figure(figsize=(10, 3))

    ### gt
    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(target_image.squeeze(), "gray")
    ax.axis("off")
    ax.set_title(f"Target", fontweight="bold")

    for t_idx, t in enumerate(timesteps):
        ax = fig.add_subplot(2, 5, t_idx + 2)
        stim_pred = F.interpolate(
            torch.from_numpy(imgs[t]).unsqueeze(0), size=(36, 64), mode="bilinear", align_corners=False
        )[0]
        ax.imshow(crop(stim_pred, crop_win).squeeze(), "gray")
        ax.set_title(f"t={t}")
        ax.axis("off")

    if show:
        plt.show()

    if save_to is not None:
        fig.savefig(save_to)

    plt.close(fig)


class GaussianBlur:
    """Blur an image with a Gaussian window.
    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
        mei_only (True/False): for transparent mei, if True, no Gaussian blur for transparent channel:
            default should be False (also for non transparent case)
    """

    def __init__(
        self, sigma, decay_factor=None, truncate=4, pad_mode="reflect", mei_only=False
    ):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.mei_only = mei_only

    def __call__(self, x, iteration=None):

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        if self.mei_only:
            num_channels = x.shape[1] - 1
            padded_x = F.pad(
                x[:, :-1, ...],
                pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                mode=self.pad_mode,
            )
        else:  # also blur transparent channel
            num_channels = x.shape[1]
            padded_x = F.pad(
                x,
                pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                mode=self.pad_mode,
            )
        blurred_x = F.conv2d(
            padded_x,
            y_gaussian.repeat(num_channels, 1, 1)[..., None],
            groups=num_channels,
        )
        blurred_x = F.conv2d(
            blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels
        )
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        # print(final_x.shape)
        if self.mei_only:
            return torch.cat(
                (final_x, x[:, -1, ...].view(x.shape[0], 1, x.shape[2], x.shape[3])),
                dim=1,
            )
        else:
            return final_x

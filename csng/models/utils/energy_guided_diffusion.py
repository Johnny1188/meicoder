import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy import signal
from tqdm import tqdm
import time
from functools import partial

from csng.utils.data import standardize, normalize, crop
from egg.diffusion import EGG


class EGGDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        egg_model_cfg,
        crop_win,
        encoder_input_shape,
        energy_scale=1,
        energy_constraint=60,
        num_steps=1000,
        energy_freq=1,
        device="cuda",
    ):
        super().__init__()
        self.encoder = encoder
        self.egg_model = EGG(**egg_model_cfg).to(device)
        self.encoder.eval()
        
        self.crop_win = crop_win
        self.encoder_input_shape = encoder_input_shape
        self.energy_scale = energy_scale
        self.energy_constraint = energy_constraint
        self.num_steps = num_steps
        self.energy_freq = energy_freq
        self.device = device

    def forward(self, resp, data_key=None, neuron_coords=None, pupil_center=None):
        self.energy_history, stim_pred, self.stim_pred_history = do_run(
            model=self.egg_model,
            energy_fn=partial(
                energy_fn,
                encoder_model=partial(self.encoder, data_key=data_key, pupil_center=pupil_center),
                encoder_input_shape=self.encoder_input_shape,
                target_response=resp,
                norm=self.energy_constraint,
                em_weight=1,
                crop_win=self.crop_win,
                energy_freq=self.energy_freq,
            ),
            energy_scale=self.energy_scale,
            num_timesteps=self.num_steps,
            num_samples=resp.shape[0],
            stim_shape=self.encoder_input_shape,
            grayscale=True,
            init_imgs=None,
            approximate_xstart_for_energy=True,
        )

        return stim_pred.detach()


def do_run(
    model,
    energy_fn,
    energy_scale,
    num_timesteps,
    num_samples=1,
    stim_shape=(36, 64),
    progressive=True,
    desc="progress",
    grayscale=True,
    init_imgs=None,
    approximate_xstart_for_energy=True
):
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
        energy_history.append(energy_fn(samples_t["pred_xstart"], t=0)["train"].detach().item())
        stim_pred_history.append(samples_t["pred_xstart"].mean(dim=1, keepdim=True).detach().cpu().numpy())

    stim_pred = F.interpolate(
        samples_t["pred_xstart"].detach(),
        size=stim_shape,
        mode="bilinear",
        align_corners=False,
    ).mean(dim=1, keepdim=True)

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
        encoder_input_shape=(36, 64),
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
        x.clone(), size=encoder_input_shape, mode="bilinear", align_corners=False
    ).mean(1, keepdim=True)
    tar = tar / torch.norm(tar, dim=(2,3), keepdim=True) * norm
    resp_pred = encoder_model(tar)
    energy += em_weight * torch.mean((resp_pred - target_response) ** 2, dim=1).sum()

    ### decoder matching
    assert xs_zero_to_match is None or len(xs_zero_to_match) == 0, "xs_zero_to_match not implemented yet"
    # tar = F.interpolate(
    #     x.clone(), size=crop_win, mode="bilinear", align_corners=False
    # ).mean(1, keepdim=True)
    # tar = crop(tar, crop_win)
    # tar = tar / torch.norm(tar, dim=(2,3), keepdim=True) * norm
    # for x_zero_to_match in xs_zero_to_match:
    #     energy += dm_weight * dm_loss_fn(tar, x_zero_to_match)

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
            torch.from_numpy(imgs[t]).unsqueeze(0),
            size=(36, 64),
            mode="bilinear",
            align_corners=False
        )[0]
        ax.imshow(crop(stim_pred, crop_win).squeeze(), "gray")
        ax.set_title(f"t={t}")
        ax.axis("off")

    if show:
        plt.show()

    if save_to is not None:
        fig.savefig(save_to)

    plt.close(fig)

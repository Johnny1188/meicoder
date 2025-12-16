import os
import random
import numpy as np
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
import torch.nn.functional as F

from csng.utils.mix import update_config_paths
from csng.utils.data import crop, standardize, normalize
from csng.models.readins import MultiReadIn


def eval_decoder(
    model,
    dataloaders,
    loss_fns,
    crop_wins,
    max_batches=None,
    eval_every_n_samples=None,
    z_score_wrt_target=False,
    device=None,
):
    assert "total" not in loss_fns, "Please provide loss functions for each data key separately"
    model.eval()

    ### for tracking over whole dataset (or mini-batches)
    losses = {data_key: {loss_fn_name: 0 for loss_fn_name in data_key_loss_fns.keys()} for data_key, data_key_loss_fns in loss_fns.items()}
    n_samples_counter = {data_key: 0 for data_key in loss_fns.keys()}
    n_samples_counter_total = {data_key: 0 for data_key in loss_fns.keys()}
    preds, targets, resps = defaultdict(list), defaultdict(list), defaultdict(list)

    def update_evals(batch_preds, batch_targets, batch_resps, batch_data_key):
        ### calculate metrics (mini-batched)
        assert batch_data_key != "total", "Data key cannot be 'total'"
        batch_preds = torch.cat(batch_preds, dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)
        batch_resps = torch.cat(batch_resps, dim=0)
        assert (B := n_samples_counter[batch_data_key]) == batch_preds.shape[0] == batch_targets.shape[0] == batch_resps.shape[0], \
            "Number of samples in preds and targets must be the same"

        for loss_name, loss_fn in loss_fns[batch_data_key].items():
            if z_score_wrt_target:
                batch_preds = (batch_preds - batch_preds.mean((1,2,3), keepdim=True)) / batch_preds.std((1,2,3), keepdim=True) * batch_targets.std((1,2,3), keepdim=True) + batch_targets.mean((1,2,3), keepdim=True)
                batch_targets = (batch_targets - batch_targets.mean((1,2,3), keepdim=True)) / batch_targets.std((1,2,3), keepdim=True) * batch_targets.std((1,2,3), keepdim=True) + batch_targets.mean((1,2,3), keepdim=True)
            losses[batch_data_key][loss_name] += loss_fn(
                batch_preds,
                batch_targets,
                resp=batch_resps,
                data_key=batch_data_key,
                sum_over_samples=False,
                phase="val",
            ).item() * B
        n_samples_counter_total[batch_data_key] += B

    ### run eval
    for k, dl in dataloaders.items(): # different data sources (cat_v1, mouse_v1, ...)
        for b_idx, b in enumerate(dl):
            ### combine losses from all data keys
            for dp in b:
                if device is not None:
                    dp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in dp.items()}

                ### get predictions
                stim_pred = model(
                    dp["resp"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    pupil_center=dp["pupil_center"],
                )

                ### append for batched metric eval
                preds[dp["data_key"]].append(crop(stim_pred, crop_wins[dp["data_key"]]).detach().cpu())
                targets[dp["data_key"]].append(crop(dp["stim"], crop_wins[dp["data_key"]]).cpu())
                resps[dp["data_key"]].append(dp["resp"].cpu())
                n_samples_counter[dp["data_key"]] += dp["stim"].shape[0]

                ### eval metrics
                if eval_every_n_samples and n_samples_counter[dp["data_key"]] >= eval_every_n_samples:
                    update_evals(batch_preds=preds[dp["data_key"]], batch_targets=targets[dp["data_key"]], batch_resps=resps[dp["data_key"]], batch_data_key=dp["data_key"])
                    preds[dp["data_key"]], targets[dp["data_key"]], resps[dp["data_key"]] = [], [], []
                    n_samples_counter[dp["data_key"]] = 0

            if max_batches is not None and b_idx + 1 >= max_batches:
                break

    ### final evaluation of metrics + aggregation across data keys
    losses["total"] = defaultdict(float)
    for data_key in preds.keys():
        if len(preds[data_key]) > 0:
            update_evals(batch_preds=preds[data_key], batch_targets=targets[data_key], batch_resps=resps[data_key], batch_data_key=data_key)

        for loss_name, loss_fn in loss_fns[data_key].items():
            losses[data_key][loss_name] /= n_samples_counter_total[data_key]
            losses["total"][loss_name] += losses[data_key][loss_name] / len(preds.keys())

    return losses


def find_best_ckpt(get_dl_fn, config, ckpt_paths, metrics):
    best_ckpt_path, best_loss = None, np.inf

    for ckpt_path in ckpt_paths:
        # decoder, _ = load_decoder_from_ckpt(config=config, ckpt_path=ckpt_path, load_best=False, load_only_core=False)
        decoder, _ = load_decoder_from_ckpt(ckpt_path=ckpt_path, device=config["device"], load_best=False, load_only_core=False, model_init_dict=None, strict=True)
        decoder.eval()
        
        ### eval
        val_dl = get_dl_fn()
        val_loss = eval_decoder(
            model=decoder,
            dataloaders=val_dl,
            loss_fns={data_key: {config["comparison"]["find_best_ckpt_according_to"]: metrics[data_key][config["comparison"]["find_best_ckpt_according_to"]]} for data_key in metrics.keys()},
            crop_wins=config["crop_wins"],
            # calc_fid="fid" in config["comparison"]["find_best_ckpt_according_to"].lower(),
        )["total"][config["comparison"]["find_best_ckpt_according_to"]]

        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt_path = ckpt_path

    return best_ckpt_path, best_loss


def load_decoder_from_ckpt(ckpt_path, device, load_best=False, load_only_core=False, model_init_dict=None, strict=True, update_paths=True):
    ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)

    if update_paths:
        ckpt["config"] = update_config_paths(ckpt["config"], os.environ["DATA_PATH"])

    if model_init_dict is not None:
        decoder = MultiReadIn(**model_init_dict).to(device)
    else:
        decoder = MultiReadIn(**ckpt["config"]["decoder"]["model"]).to(device)

    decoder.load_from_ckpt(ckpt=ckpt, load_best=load_best, load_only_core=load_only_core, strict=strict)

    return decoder, ckpt


def collect_all_preds_and_targets(model, dataloaders, crop_wins, device=None):
    preds, targets = defaultdict(list), defaultdict(list)

    for k, dl in dataloaders.items(): # different data sources (cat_v1, mouse_v1, ...)
        for b_idx, b in enumerate(dl):
            for dp in b:
                if device is not None:
                    dp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in dp.items()}

                ### get predictions
                stim_pred = model(
                    dp["resp"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    pupil_center=dp["pupil_center"],
                )

                ### save preds and targets
                preds[dp["data_key"]].append(crop(stim_pred, crop_wins[dp["data_key"]]).detach().cpu())
                targets[dp["data_key"]].append(crop(dp["stim"], crop_wins[dp["data_key"]]).cpu())

    ### concatenate all preds and targets
    for data_key in preds.keys():
        preds[data_key] = torch.cat(preds[data_key], dim=0)
        targets[data_key] = torch.cat(targets[data_key], dim=0)
    preds, targets = dict(preds), dict(targets)

    return preds, targets


class SavedReconstructionsDecoder:
    def __init__(self, reconstructions, data_key, zscore_reconstructions=False, device="cuda"):
        self.recons = reconstructions
        self.data_key = data_key
        self.zscore_preds = zscore_reconstructions
        self.device = device

        if zscore_reconstructions:
            self.recons = normalize(self.recons)
        self.recon_idx = 0

    def reset_counter(self):
        self.recon_idx = 0

    def eval(self):
        pass

    def __call__(self, resp, data_key=None, neuron_coords=None, pupil_center=None):
        assert data_key is None or data_key == self.data_key, "Data key must be the same as the one used for obtaining the reconstructions"
        assert self.recon_idx + resp.shape[0] <= len(self.recons), "Not enough reconstructions"

        recons = self.recons[self.recon_idx:self.recon_idx+resp.shape[0]]
        self.recon_idx += resp.shape[0]

        return recons.to(self.device)


##### Plotting utils #####
def autolabel(ax, rects, fontsize=15, bold=False):
    """Attach a text label above each bar in *rects*, displaying its height.
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            rotation=90,
        )


def plot_reconstructions(runs, stim, stim_label="Target", data_key=None, manually_standardize=False, crop_win=None, save_to=None):
    fig = plt.figure(figsize=(1.5 + int(len(runs) * 1.3), int(stim.shape[0] - 2)))
    grid = ImageGrid(fig, 111, nrows_ncols=(stim.shape[0], 1 + len(runs)), direction="column", axes_pad=0.03, share_all=True)
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    def plot_imgs(imgs, curr_ax_idx):
        imgs_to_show = imgs if not manually_standardize else standardize(imgs, dim=(1,2,3))
        for img_to_show in imgs_to_show:
            grid[curr_ax_idx].imshow(img_to_show.permute(1,2,0), "gray")
            for d in ("top", "right", "left", "bottom"):
                grid[curr_ax_idx].spines[d].set_visible(False)
            curr_ax_idx += 1
        return curr_ax_idx

    def set_title(ax, title):
        ax.set_title(title, fontsize=8, rotation=90, va="baseline")

    ### plot stim
    ax_idx = 0
    set_title(ax=grid[ax_idx], title=stim_label)
    imgs = stim.cpu()
    if crop_win is not None:
        imgs = crop(imgs, crop_win)
    ax_idx = plot_imgs(imgs, curr_ax_idx=ax_idx)

    ### plot other
    for run_name in runs:
        set_title(ax=grid[ax_idx], title=run_name)
        imgs = runs[run_name]["stim_pred_best"][0].cpu() if data_key is None else runs[run_name]["stim_pred_best"][0][data_key].cpu()
        if crop_win is not None:
            imgs = crop(imgs, crop_win)
        ax_idx = plot_imgs(imgs, curr_ax_idx=ax_idx)

    plt.show()

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight")

    plt.close(fig)


def plot_metrics(runs_to_compare, losses_to_plot, bar_width=0.8, save_to=None):
    sns.set_style("whitegrid")
    c_palette = sns.color_palette("tab10", n_colors=len(losses_to_plot))

    num_methods = len(runs_to_compare)
    num_metrics = len(losses_to_plot)

    fig_width = max(5, num_methods * (num_metrics // 2 + 1))
    fig_height = max(5, num_metrics * 1.65)

    k = list(runs_to_compare.keys())[0]
    for run_idx in range(len(runs_to_compare[k]["test_losses"])):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        index = np.arange(num_methods)
        bar_spacing = bar_width / num_metrics

        for i, (method, run_dict) in enumerate(runs_to_compare.items()):
            for j, loss_name in enumerate(losses_to_plot):
                value = run_dict["test_losses"][run_idx]["total"].get(loss_name, 0)
                rects = ax.bar(
                    i - bar_width / 2 + j * bar_spacing,
                    value,
                    width=bar_spacing,
                    color=c_palette[j],
                    label=loss_name if i == 0 else ""
                )

                # min_loss = min(
                #     runs_to_compare[_k]["test_losses"][run_idx]["total"].get(loss_name, float('inf'))
                #     for _k in runs_to_compare.keys()
                # )
                # is_lowest = value == min_loss
                # autolabel(ax, rects, fontsize=12, bold=is_lowest)
                autolabel(ax, rects, fontsize=12, bold=False)

        ax.set_xticks(index)
        ax.set_xticklabels(runs_to_compare.keys(), rotation=45, ha="right")
        ax.set_xlabel("Method", fontsize=14)
        # ax.set_ylabel("Value", fontsize=14)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=num_metrics, frameon=False, fontsize=12)

        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(False)
        sns.despine()
        plt.tight_layout()
        plt.show()

        if save_to:
            fig.savefig(save_to, bbox_inches="tight")

        plt.close(fig)

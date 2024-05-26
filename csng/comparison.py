import os
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
import torch.nn.functional as F
from focal_frequency_loss import FocalFrequencyLoss as FFL

from csng.utils import crop, standardize, normalize, dict_to_str
from csng.readins import MultiReadIn
from csng.losses import (
    MultiSSIMLoss,
    SSIMLoss,
    SSIM,
    CroppedLoss,
    Loss,
    MS_SSIMLoss,
    PerceptualLoss,
    EncoderPerceptualLoss,
    VGGPerceptualLoss,
)


def get_metrics(crop_win=None, device="cpu"):
    metrics = {
        "SSIM": Loss(config=dict(
            loss_fn=SSIM(),
            window=crop_win,
            standardize=True,
        )),
        "Log SSIML": Loss(config=dict(
            loss_fn=SSIMLoss(
                log_loss=True,
                inp_normalized=True,
                inp_standardized=False,
            ),
            window=crop_win,
        )),
        "SSIML": Loss(config=dict(
            loss_fn=SSIMLoss(
                log_loss=False,
                inp_normalized=True,
                inp_standardized=False,
            ),
            window=crop_win,
        )),
        "PL": Loss(config=dict(
            loss_fn=VGGPerceptualLoss(
                resize=False,
                device=device,
            ),
            window=crop_win,
            standardize=True,
        )),
        "FFL": Loss(config=dict(
            loss_fn=FFL(loss_weight=1, alpha=1.0),
            window=crop_win,
            standardize=True,
        )),
        "MSE": Loss(config=dict(
            loss_fn=lambda x_hat, x: F.mse_loss(
                standardize(crop(x_hat, crop_win)),
                standardize(crop(x, crop_win)),
                reduction="none",
            ).mean((1,2,3)).sum(),
            window=crop_win,
        )),
        "MSE-no-standardization": Loss(config=dict(
            loss_fn=lambda x_hat, x: F.mse_loss(
                crop(x_hat, crop_win),
                crop(x, crop_win),
                reduction="none",
            ).mean((1,2,3)).sum(),
            window=crop_win,
        )),
        "MAE": Loss(config=dict(
            loss_fn=lambda x_hat, x: F.l1_loss(
                standardize(crop(x_hat, crop_win)),
                standardize(crop(x, crop_win)),
                reduction="none",
            ).mean((1,2,3)).sum(),
            window=crop_win,
        )),
    }
    metrics["SSIML-PL"] = Loss(config=dict(
        loss_fn=lambda y_hat, y, **kwargs: metrics["SSIML"](y_hat, y, **kwargs) + metrics["PL"](y_hat, y, **kwargs),
        window=crop_win,
    ))

    return metrics


def load_decoder_from_ckpt(ckpt_path, device, load_best=False):
    ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
    ckpt_config = ckpt["config"]

    ### TODO: remove (quick fix) -->
    if "meis_path" in ckpt_config["decoder"]["model"]["readins_config"][-1]["layers"][0][1] \
        and not os.path.exists(ckpt_config["decoder"]["model"]["readins_config"][-1]["layers"][0][1]["meis_path"]):
        # and "/home/sobotj11/decoding-brain-activity/data/mouse_v1_sensorium22/meis/" in ckpt_config["decoder"]["model"]["readins_config"][-1]["layers"][0][1]["meis_path"]:
        for rc in ckpt_config["decoder"]["model"]["readins_config"]:
            rc["layers"][0][1]["meis_path"] = rc["layers"][0][1]["meis_path"].replace(
                "/media/jsobotka/ext_ssd/csng_data/mouse_v1_sensorium22/meis/",
                "/home/sobotj11/decoding-brain-activity/data/mouse_v1_sensorium22/meis/",
            )
    ### TODO: remove (quick fix) <--

    decoder = MultiReadIn(**ckpt_config["decoder"]["model"]).to(device)
    if load_best:
        decoder._load_state_dict(ckpt["best"]["model"])
    else:
        decoder._load_state_dict(ckpt["decoder"])
    decoder.eval()

    return decoder, ckpt


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


def plot_reconstructions(runs, stim, stim_label="Target", manually_standardize=False, crop_win=None, save_to=None):
    fig = plt.figure(figsize=(10, 6))
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
        imgs = runs[run_name]["stim_pred_best"][0].cpu()
        if crop_win is not None:
            imgs = crop(imgs, crop_win)
        ax_idx = plot_imgs(imgs, curr_ax_idx=ax_idx)

    plt.show()

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight")


def plot_metrics(runs_to_compare, losses_to_plot, bar_width=0.7, save_to=None):
    plt.rc("pdf", fonttype=42)
    plt.rc("ps", fonttype=42)
    plt.rc("legend", fontsize=13)
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    plt.rc("axes", labelsize=13)
    plt.rc("axes", titlesize=13)
    plt.rc("axes", linewidth=0.5)
    plt.rc("axes", labelpad=10)
    plt.rc("lines", linewidth=1.)
    c_palette = list(plt.cm.tab10.colors)
    plt.rc("axes", prop_cycle=plt.cycler("color", c_palette))
    plt.rc("figure", dpi=300)
    plt.rc("savefig", dpi=300)
    plt.rc("savefig", format="pdf")
    plt.rc("savefig", bbox="tight")
    plt.rc("savefig", pad_inches=0.1)

    ### plot
    k = list(runs_to_compare.keys())[0]
    for run_idx in range(len(runs_to_compare[k]["test_losses"])):
        ### bar plot of test losses
        fig = plt.figure(figsize=(22, 7))
        # fig = plt.figure(figsize=(55, 15))
        ax = fig.add_subplot(111)
        print(runs_to_compare[k]["ckpt_paths"][run_idx])

        ### grouped bar plot
        for i, (k, run_dict) in enumerate(runs_to_compare.items()):
            for j, loss in enumerate(losses_to_plot):
                rects = ax.bar(
                    i - bar_width / len(losses_to_plot) + j * bar_width / len(losses_to_plot),
                    run_dict["test_losses"][run_idx][loss]["total"],
                    width=bar_width / len(losses_to_plot),
                    color=c_palette[j],
                )
                lowest_loss_flag = False
                if run_dict["test_losses"][run_idx][loss]["total"] == min(
                    [runs_to_compare[_k]["test_losses"][run_idx][loss]["total"] for _k in runs_to_compare.keys()]
                ):
                    lowest_loss_flag = True
                autolabel(ax=ax, rects=rects, fontsize=18, bold=lowest_loss_flag)

        ### add legend with color explanation
        ax.legend(
            handles=[
                mpatches.Patch(color=c_palette[i], label=loss)
                for i, loss in enumerate(losses_to_plot)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=len(losses_to_plot),
            fontsize=20,
            frameon=False,
        )

        ax.set_xticks(range(len(runs_to_compare)))
        ax.set_xticklabels(runs_to_compare.keys())
        ### with rotatation of the xtick labels
        ax.set_xticklabels(
            [k for k in runs_to_compare.keys()],
            rotation=90,
            y=-0.7,
            ha="right",
            va="baseline",
        )
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_ylim(0, None)

        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlabel("Decoder", fontsize=22, labelpad=40)
        ax.set_ylabel("Loss", fontsize=22, labelpad=40)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.show()

        if save_to is not None:
            fig.savefig(save_to, bbox_inches="tight")


def plot_syn_data_loss_curve(run_groups, losses_to_plot, run_group_colors=None, mean_line_kwargs=None, xlabel="% of synthetic data", save_to=None):
    fig, ax = plt.subplots(ncols=len(losses_to_plot), figsize=(11, 3.5))

    all_group_losses, all_group_syn_data_percentages = {l_name: [] for l_name in losses_to_plot}, []
    for (run_g_name, runs), g_color in zip(run_groups.items(), run_group_colors if run_group_colors is not None else [None]*len(run_groups)):
        ### collect x and y values
        losses, syn_data_percentages = {l_name: [] for l_name in losses_to_plot}, []
        for run in runs.values():
            ### get losses
            if len(run["test_losses"]) > 1:
                print(f"[WARNING] Length to the 'test_losses' list is more than 1. Taking only the last loss!")
            for l_name in losses_to_plot:
                losses[l_name].append(run["test_losses"][-1][l_name]["total"])

            ### compute syn_data_percentage
            run_config = run["configs"][-1]
            base_data_batch_size = run_config["data"]["mouse_v1"]["dataset_config"]["batch_size"] \
                if not run_config["data"]["mouse_v1"]["skip_train"] else 0
            if "syn_dataset_config" in run_config["data"] \
                and run_config["data"]["syn_dataset_config"] is not None \
                and run_config["data"]["syn_dataset_config"]["batch_size"] > 0:
                total_batch_size = run_config["data"]["syn_dataset_config"]["batch_size"] + base_data_batch_size
                syn_data_percentage = run_config["data"]["syn_dataset_config"]["batch_size"] / total_batch_size
                syn_data_percentage = np.round(syn_data_percentage * 100, 2)
            else:
                syn_data_percentage = 0
            syn_data_percentages.append(syn_data_percentage)
        
        all_group_syn_data_percentages.append(syn_data_percentages)
        for l_name in all_group_losses:
            all_group_losses[l_name].append(losses[l_name])
        
        ### plot
        for l_idx, (l_name, l_vals) in enumerate(losses.items()):
            plot_kwargs = {
                "linestyle": "--",
                "linewidth": 2,
                "label": run_g_name,
            }
            if g_color is not None: plot_kwargs["color"] = g_color
            ax[l_idx].plot(syn_data_percentages, l_vals, **plot_kwargs)

    ### plot mean
    if mean_line_kwargs:
        assert np.all([all_group_syn_data_percentages[i] == all_group_syn_data_percentages[i+1] for i in range(len(all_group_syn_data_percentages) - 1)])
        for group_losses, _ax in zip(all_group_losses.values(), ax):
            _ax.plot(all_group_syn_data_percentages[0], np.array(group_losses).mean(0), **mean_line_kwargs)


    ### styling
    for i, (l_name, _ax) in enumerate(zip(losses_to_plot, ax)):
        _ax.locator_params(axis="x", nbins=3)
        _ax.set_xticks(_ax.get_xticks(), fontsize=12)
        _ax.set_yticks(_ax.get_yticks(), fontsize=12)
        _ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, min_n_ticks=2))
        _ax.set_xlim(-5, 105)
        _ax.set_xlabel(xlabel, labelpad=15, fontsize=13)
        _ax.set_ylabel(l_name, labelpad=15, fontsize=13)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.grid(visible=True, which="major", axis="y", alpha=0.5)
        
        _ax.legend()
        if i == len(ax) - 1:
            handles, labels = _ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=[0.5, 1.12], ncol=5, frameon=False, fontsize=13)
        _ax.get_legend().remove()
    
    fig.tight_layout(w_pad=3)
    plt.show()
    
    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight")

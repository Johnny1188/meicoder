import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import json
import pandas as pd
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt
import wandb
from nnfabrik.builder import get_data
from focal_frequency_loss import FocalFrequencyLoss as FFL

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import crop, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, plot_losses
from csng.losses import (
    MultiSSIMLoss,
    SSIMLoss,
    CroppedLoss,
    Loss,
    MS_SSIMLoss,
    PerceptualLoss,
    EncoderPerceptualLoss,
    VGGPerceptualLoss,
)
from csng.data import MixedBatchLoader

from BoostedInvertedEncoder import BoostedInvertedEncoder
from encoder import get_encoder
from data_utils import get_mouse_v1_data, PerSampleStoredDataset, append_syn_dataloaders, append_data_aug_dataloaders

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
print(f"{DATA_PATH=}")


def eval_decoder(model, dataloader, loss_fns, normalize_decoded, config):
    model.eval()
    val_losses = {loss_fn_name: {"total": 0} for loss_fn_name in loss_fns.keys()}
    num_samples = 0
    denom_data_keys = {}

    for b in dataloader:
        ### combine from all data keys
        for data_key, stim, resp, neuron_coords, pupil_center in b:
            if model.__class__.__name__ == "InvertedEncoder":
                stim_pred, _, _ = model(
                    resp_target=resp,
                    stim_target=stim,
                    additional_encoder_inp={
                        "data_key": data_key,
                        "pupil_center": pupil_center,
                    }
                )
            elif hasattr(model, "core") and model.core.__class__.__name__ == "L2O_Decoder":
                raise NotImplementedError("L2O_Decoder not implemented yet - data needs to be standardized")
                stim_pred, _ = model(
                    x=resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                    additional_core_inp=dict(
                        train=False,
                        stim=None,
                        resp=resp,
                        neuron_coords=neuron_coords,
                        pupil_center=pupil_center,
                        data_key=data_key,
                        n_steps=config["decoder"]["n_steps"],
                        x_hat_history_iters=None,
                    ),
                )
            elif isinstance(model, BoostedInvertedEncoder):
                stim_pred, _, _ = model(
                    resp,
                    train=False,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                )
            else:
                stim_pred = model(
                    resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                )

            if normalize_decoded:
                stim_pred = normalize(stim_pred)

            for loss_fn_name, loss_fn in loss_fns.items():
                loss = loss_fn(stim_pred, stim, data_key=data_key, phase="val").item()
                val_losses[loss_fn_name]["total"] += loss
                val_losses[loss_fn_name][data_key] = loss if data_key not in val_losses[loss_fn_name] else val_losses[loss_fn_name][data_key] + loss
            
            num_samples += stim.shape[0]
            denom_data_keys[data_key] = denom_data_keys[data_key] + stim.shape[0] if data_key in denom_data_keys else stim.shape[0]

    for loss_name in val_losses:
        val_losses[loss_name]["total"] /= num_samples
        for k in denom_data_keys:
            val_losses[loss_name][k] /= denom_data_keys[k]

    return val_losses


def get_all_data(config):
    dls, neuron_coords = get_mouse_v1_data(config=config["data"])
    if "syn_dataset_config" in config["data"] and config["data"]["syn_dataset_config"] is not None:
        dls = append_syn_dataloaders(dls, config=config["data"]["syn_dataset_config"]) # append synthetic data
    if "data_augmentation" in config["data"] and config["data"]["data_augmentation"] is not None:
        dls = append_data_aug_dataloaders(
            dataloaders=dls,
            config=config["data"]["data_augmentation"],
        )
    return dls, neuron_coords


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
            xytext=(0, 10),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            rotation=90,
        )


def dict_to_str(d):
    return ", ".join([f"{k}: {v:.3f}" for k, v in d.items()])


def plot_reconstructions(runs_to_compare, stim, config, loss_fns, sample_data_key):
    ### plot reconstructions
    for k in runs_to_compare.keys():
        for run_idx in range(len(runs_to_compare[k]["stim_pred_best"])):
            # print(k, "\n", run_dict["ckpt_paths"][run_idx])
            stim_pred = runs_to_compare[k]["stim_pred_best"][run_idx]
            recon_losses = dict()
            for loss_fn_name, loss_fn in loss_fns.items():
                recon_losses[loss_fn_name] = loss_fn(
                    stim_pred[:8].to(config["device"]), stim[:8].to(config["device"]), data_key=sample_data_key, phase="val"
                ).item() / stim_pred[:8].shape[0]
                if not (
                    "VGG" in loss_fn_name \
                    or "FFL" in loss_fn_name \
                    or "Log" in loss_fn_name \
                    or loss_fn_name == "MSE" \
                    or loss_fn_name == "MAE"
                ):
                    del recon_losses[loss_fn_name]
            print(dict_to_str(recon_losses))
            fig = plot_comparison(
                target=crop(stim[:8], config["crop_win"]).cpu(),
                pred=crop(stim_pred[:8], config["crop_win"]).cpu(),
            )


def plot_img_at_ax(ax, img):
    ax.imshow(img.cpu().squeeze(), cmap="gray")
    plt.xticks([])
    plt.yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return ax


def plot_reconstructions_publication(runs_to_compare, stim, config, save_to=None):
    fig = plt.figure(facecolor="white")

    img_grid_shape = (stim.shape[0], 1 + len(runs_to_compare))

    ### plot comparison
    for row_i in range(img_grid_shape[0]):
        ### plot target
        ax = fig.add_subplot(
            img_grid_shape[0],
            img_grid_shape[1],
            1 + row_i*img_grid_shape[1],
        )
        ax = plot_img_at_ax(ax, crop(stim[row_i], config["crop_win"]))
        if row_i == 0:
            ax.set_title("Target", fontsize=5.5, fontweight="bold", rotation = 45, va='baseline')
        for col_j, k in enumerate(runs_to_compare.keys()):
            for run_idx in range(len(runs_to_compare[k]["stim_pred_best"])):
                ax = fig.add_subplot(
                    img_grid_shape[0],
                    img_grid_shape[1],
                    1 + row_i*img_grid_shape[1] + 1 + col_j,
                )
                stim_pred = runs_to_compare[k]["stim_pred_best"][run_idx][row_i]
                ax = plot_img_at_ax(ax, crop(stim_pred, config["crop_win"]))
                if row_i == 0:
                    ax.set_title(k, fontsize=5.5, rotation=45, va='baseline')

    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.show()

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight")


def plot_metrics(runs_to_compare, losses_to_plot, bar_width=0.7):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    ### plot
    k = list(runs_to_compare.keys())[0]
    for run_idx in range(len(runs_to_compare[k]["test_losses"])):
        ### bar plot of test losses
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)
        print(runs_to_compare[k]["ckpt_paths"][run_idx])

        ### grouped bar plot
        for i, (k, run_dict) in enumerate(runs_to_compare.items()):
            for j, loss in enumerate(losses_to_plot):
                rects = ax.bar(
                    i - bar_width / len(losses_to_plot) + j * bar_width / len(losses_to_plot),
                    run_dict["test_losses"][run_idx][loss]["total"],
                    width=bar_width / len(losses_to_plot),
                    color=colors[j],
                )
                autolabel(ax=ax, rects=rects)

        ### add legend with color explanation
        from matplotlib import patches as mpatches
        ax.legend(
            handles=[
                mpatches.Patch(color=colors[i], label=loss)
                for i, loss in enumerate(losses_to_plot)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.28),
            ncol=len(losses_to_plot),
            fontsize=14,
            frameon=False,
        )

        ax.set_title(
            "Test Losses",
            fontsize=18,
            pad=90,
        )
        ax.set_xticks(range(len(runs_to_compare)))
        ax.set_xticklabels(runs_to_compare.keys())
        ### with rotatation of the xtick labels
        ax.set_xticklabels(
            [k for k in runs_to_compare.keys()],
            rotation=15,
            ha="right",
        )
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlabel("Decoder", fontsize=14, labelpad=20)
        ax.set_ylabel("Loss", fontsize=14, labelpad=20)
        ax.set_ylim(0, None)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.show()


def plot_over_training(runs_to_compare, to_plot="val_loss", conv_win=10, ckpt_idx=0):
    ### plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for k, run_dict in runs_to_compare.items():
        if run_dict["history"][ckpt_idx] is None or to_plot not in run_dict["history"][ckpt_idx]:
            print(f"Skipping {k}...")
            continue
        if conv_win is not None and (run_dict["history"][ckpt_idx] is not None and np.nan not in run_dict["history"][ckpt_idx]):
            vals_to_plot = np.convolve(run_dict["history"][ckpt_idx][to_plot], np.ones(conv_win) / conv_win, mode="valid")
        else:
            vals_to_plot = run_dict["history"][ckpt_idx][to_plot]
        ax.plot(
            [t for t in range(len(vals_to_plot)) if vals_to_plot[t] is not np.nan],
            [v for v in vals_to_plot if v is not np.nan],
            label=k,
            linewidth=3,
        )

    if to_plot == "train_loss":
        ax.set_title("Training log SSIM loss", fontsize=16, pad=20)
    elif to_plot == "val_loss":
        ax.set_title("Validation log SSIM loss", fontsize=16, pad=20)
    else:
        raise ValueError(f"Unknown loss type: {to_plot}")

    ax.set_xlabel("Epoch", fontsize=15, labelpad=20)
    ax.set_ylabel("Log SSIM loss", fontsize=15, labelpad=20)
    # ax.set_ylim(1.25, None)
    # ax.set_ylim(1.3, 1.75)
    # ax.set_xlim(0, 80)
    ax.legend(
        loc="upper right",
        # loc="upper center",
        # loc="lower left",
        # loc="lower center",
        fontsize=14,
        frameon=False,
        # bbox_to_anchor=(1.16, 1),
        bbox_transform=ax.transAxes,
        # title="",
        title_fontsize=15,
        ncol=1,
    )
    # increase width of legend lines
    leg = ax.get_legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)


    # set larger font for x and y ticks
    ax.tick_params(axis="both", which="major", labelsize=14)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


def plot_metrics_publication(runs_to_compare, losses_to_plot, bar_width=0.7, save_to=None):
    # colors = [
    #     "#1f77b4",
    #     "#ff7f0e",
    #     "#2ca02c",
    #     "#d62728",
    #     "#9467bd",
    #     "#8c564b",
    #     "#e377c2",
    #     "#7f7f7f",
    #     "#bcbd22",
    # ]
    # plt.rc("font", family="Times New Roman")
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
    # plt.rc("figure", figsize=(6, 4))
    # plt.rc("figure", figsize=(2.5, 2.5))
    plt.rc("savefig", dpi=300)
    plt.rc("savefig", format="pdf")
    plt.rc("savefig", bbox="tight")
    plt.rc("savefig", pad_inches=0.1)

    ### plot
    k = list(runs_to_compare.keys())[0]
    for run_idx in range(len(runs_to_compare[k]["test_losses"])):
        ### bar plot of test losses
        fig = plt.figure(figsize=(20, 7))
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
            rotation=0,
            y=-0.03,
            # ha="right",
            # va="baseline",
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

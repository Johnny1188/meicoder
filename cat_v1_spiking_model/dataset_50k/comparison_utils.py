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
from focal_frequency_loss import FocalFrequencyLoss as FFL
import lovely_tensors as lt

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import plot_losses, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, crop
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
from csng.readins import MultiReadIn, FCReadIn, ConvReadIn

from cat_v1_spiking_model.dataset_50k.data import (
    prepare_v1_dataloaders,
    SyntheticDataset,
    BatchPatchesDataLoader,
    MixedBatchLoader,
    PerSampleStoredDataset,
)


DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
print(f"{DATA_PATH=}")


def load_decoder_from_ckpt(config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=config["device"], pickle_module=dill)
    ckpt_config = ckpt["config"]

    decoder = MultiReadIn(**ckpt_config["decoder"]["model"]).to(config["device"])
    if config["comparison"]["eval_all_ckpts"] or not config["comparison"]["load_best"]:
        decoder._load_state_dict(ckpt["decoder"])
    else:
        decoder._load_state_dict(ckpt["best"]["model"])
    decoder.eval()

    return decoder, ckpt


def get_metrics(config):
    metrics = {
        "SSIM": CroppedLoss(
            window=config["crop_win"],
            normalize=False,
            standardize=True,
            # loss_fn=SSIM(size_average=False, reduction="sum"),
            loss_fn=SSIM(reduction="sum"),
        ),
        "Log SSIML": SSIMLoss(
            window=config["crop_win"],
            log_loss=True,
            inp_normalized=True,
            inp_standardized=False,
            reduction="sum",
        ),
        # "Log MultiSSIM Loss": MultiSSIMLoss(
        #     window=config["crop_win"],
        #     log_loss=True,
        #     inp_normalized=True,
        #     inp_standardized=False,
        #     reduction="sum",
        # ),
        "SSIML": SSIMLoss(
            window=config["crop_win"],
            log_loss=False,
            inp_normalized=True,
            inp_standardized=False,
            reduction="sum",
        ),
        # "MultiSSIM Loss": MultiSSIMLoss(
        #     window=config["crop_win"],
        #     log_loss=False,
        #     inp_normalized=True,
        #     inp_standardized=False,
        #     reduction="sum",
        # ),
        "PL": CroppedLoss(
            window=config["crop_win"],
            normalize=False,
            standardize=True,
            loss_fn=VGGPerceptualLoss(
                resize=False,
                device=config["device"],
            ),
        ),
        # "Perceptual Loss (Encoder)": CroppedLoss(
        #     window=config["crop_win"],
        #     normalize=True,
        #     standardize=False,
        #     loss_fn=EncoderPerceptualLoss(
        #         encoder=encoder,
        #         device=config["device"],
        #     ),
        # ),
        "FFL": CroppedLoss(
            window=config["crop_win"],
            normalize=False,
            standardize=True,
            loss_fn=FFL(loss_weight=1, alpha=1.0),
        ),
        "MSE": lambda x_hat, x: F.mse_loss(
            standardize(crop(x_hat, config["crop_win"])),
            standardize(crop(x, config["crop_win"])),
            reduction="none",
        ).mean((1,2,3)).sum(),
        "MAE": lambda x_hat, x: F.l1_loss(
            standardize(crop(x_hat, config["crop_win"])),
            standardize(crop(x, config["crop_win"])),
            reduction="none",
        ).mean((1,2,3)).sum(),
    }
    metrics["SSIML-PL"] = CroppedLoss(
        window=config["crop_win"],
        normalize=False,
        standardize=False,
        loss_fn=lambda y_hat, y: metrics["SSIML"](y_hat, y) + metrics["PL"](y_hat, y)
    )

    for k in metrics.keys():
        metrics[k] = Loss(
            model=None,
            config={
                "loss_fn": metrics[k],
                "l1_reg_mul": 0,
                "l2_reg_mul": 0,
                "con_reg_mul": 0,
            }
        )

    return metrics


def get_dataloaders(config):
    ### get base dataloaders
    dls = prepare_v1_dataloaders(**config["data"]["cat_v1"])

    ### split and mix dataloaders
    train_dataloader = MixedBatchLoader(
        dataloaders=[dls["train"]],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1"],
        return_pupil_center=False,
    )
    val_dataloader = MixedBatchLoader(
        dataloaders=[dls["val"]],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1"],
        return_pupil_center=False,
    )
    test_dataloader = MixedBatchLoader(
        dataloaders=[dls["test"]],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1"],
        return_pupil_center=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def find_best_ckpt(config, ckpt_paths, metrics):
    best_ckpt_path, best_loss = None, np.inf
    for ckpt_path in ckpt_paths:
        decoder, _ = load_decoder_from_ckpt(config=config, ckpt_path=ckpt_path)
        decoder.eval()
        
        ### eval
        _, val_dl, _ = get_dataloaders(config=config)
        val_loss = eval_decoder(
            model=decoder,
            dataloader=val_dl,
            loss_fns={config["comparison"]["find_best_ckpt_according_to"]: metrics[config["comparison"]["find_best_ckpt_according_to"]]},
            config=config,
        )[config["comparison"]["find_best_ckpt_according_to"]]["total"]

        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt_path = ckpt_path
    return best_ckpt_path, best_loss


def eval_decoder(model, dataloader, loss_fns, config):
    model.eval()
    val_losses = {loss_fn_name: {"total": 0} for loss_fn_name in loss_fns.keys()}
    num_samples = 0
    denom_data_keys = {}

    for b in dataloader:
        ### combine from all data keys
        for data_key, (stim, resp, neuron_coords) in b.items():
            stim = stim.to(config["device"])
            resp = resp.to(config["device"])
            neuron_coords = neuron_coords.float().to(config["device"])

            ### get predictions
            if model.__class__.__name__ == "InvertedEncoder":
                stim_pred, _, _ = model(
                    resp_target=resp,
                    stim_target=stim,
                    additional_encoder_inp={
                        "data_key": data_key,
                    }
                )
            else:
                stim_pred = model(
                    resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                )

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
            ax.set_title("Target", fontsize=5.5, fontweight="bold", rotation=45, va='baseline')
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
                    ax.set_title(k, fontsize=5.5, rotation=45, va="baseline")

    plt.subplots_adjust(wspace=0., hspace=-0.85)
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
        fig = plt.figure(figsize=(22, 7))
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
            rotation=20,
            y=-0.17,
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

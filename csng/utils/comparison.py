import os
import random
import numpy as np
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches as mpatches
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
from csng.losses import FID


def eval_decoder(model, dataloaders, loss_fns, crop_wins, calc_fid=False, max_batches=None):
    model.eval()

    ### for tracking over whole dataset
    losses = {data_key: {loss_fn_name: 0 for loss_fn_name in data_key_loss_fns.keys()} for data_key, data_key_loss_fns in loss_fns.items()}
    denom_data_keys = {}
    if calc_fid:
        preds, targets = defaultdict(list), defaultdict(list)

    ### run eval
    for k, dl in dataloaders.items(): # different data sources (cat_v1, mouse_v1, ...)
        for b in dl:
            ### combine losses from all data keys
            for dp in b:
                ### get predictions
                stim_pred = model(
                    dp["resp"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    pupil_center=dp["pupil_center"],
                )

                ### calc metrics
                for loss_fn_name, loss_fn in loss_fns[dp["data_key"]].items():
                    losses[dp["data_key"]][loss_fn_name] += loss_fn(stim_pred, dp["stim"], data_key=dp["data_key"], phase="val").item()

                ### append for later fid calculation
                if calc_fid:
                    preds[dp["data_key"]].append(crop(stim_pred, crop_wins[dp["data_key"]]).detach().cpu())
                    targets[dp["data_key"]].append(crop(dp["stim"], crop_wins[dp["data_key"]]).cpu())

                denom_data_keys[dp["data_key"]] = denom_data_keys[dp["data_key"]] + dp["stim"].shape[0] if dp["data_key"] in denom_data_keys else dp["stim"].shape[0]

            if max_batches is not None and b_idx + 1 >= max_batches:
                break

    ### average losses
    losses["total"] = defaultdict(float)
    for data_key in losses:
        if data_key == "total": continue
        for loss_name in losses[data_key]:
            losses[data_key][loss_name] /= denom_data_keys[data_key]
            losses["total"][loss_name] += losses[data_key][loss_name]
    losses["total"] = {loss_name: losses["total"][loss_name] / (len(losses.keys()) - 1) for loss_name in losses["total"]}

    ### eval fid
    if calc_fid:
        losses["total"]["FID"] = 0
        for data_key in preds.keys():
            fid = FID(inp_standardized=False, device="cpu")
            losses[data_key]["FID"] = fid(
                pred_imgs=torch.cat(preds[data_key], dim=0),
                gt_imgs=torch.cat(targets[data_key], dim=0)
            )
            losses["total"]["FID"] += losses[data_key]["FID"]
        losses["total"]["FID"] /= len(preds.keys())

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
            dataloaders={"val": val_dl},
            loss_fns={config["comparison"]["find_best_ckpt_according_to"]: metrics[config["comparison"]["find_best_ckpt_according_to"]]},
            config=config,
            calc_fid="fid" in config["comparison"]["find_best_ckpt_according_to"].lower(),
            max_batches=None,
        )[config["comparison"]["find_best_ckpt_according_to"]]["total"]

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


def plot_metrics(runs_to_compare, losses_to_plot, bar_width=0.7, save_to=None):
    c_palette = list(plt.cm.tab10.colors)

    ### plot
    k = list(runs_to_compare.keys())[0]
    for run_idx in range(len(runs_to_compare[k]["test_losses"])):
        ### bar plot of test losses
        fig = plt.figure(figsize=(25 + 6 * len(runs_to_compare[k]["test_losses"]), 12))
        ax = fig.add_subplot(111)

        ### grouped bar plot
        for i, (k, run_dict) in enumerate(runs_to_compare.items()):
            for j, loss_name in enumerate(losses_to_plot):
                rects = ax.bar(
                    i - bar_width / len(losses_to_plot) + j * bar_width / len(losses_to_plot),
                    run_dict["test_losses"][run_idx]["total"][loss_name],
                    width=bar_width / len(losses_to_plot),
                    color=c_palette[j],
                )
                lowest_loss_flag = False
                if run_dict["test_losses"][run_idx]["total"][loss_name] == min(
                    [runs_to_compare[_k]["test_losses"][run_idx]["total"][loss_name] for _k in runs_to_compare.keys()]
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
        ### rotatate xtick labels
        ax.set_xticklabels(
            [k for k in runs_to_compare.keys()],
            rotation=90,
            # y=-0.7,
            ha="right",
            # va="baseline",
        )
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_ylim(0, None)

        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlabel("Decoder", fontsize=22, labelpad=40)
        ax.set_ylabel("Loss", fontsize=22, labelpad=40)

        ### remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.show()

        if save_to is not None:
            fig.savefig(save_to, bbox_inches="tight")

        plt.close(fig)


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
            if run_config["data"]["mouse_v1"]["skip_train"]:
                base_data_batch_size == 0
            else:
                base_data_batch_size = run_config["data"]["mouse_v1"]["dataset_config"]["batch_size"] * len(run_config["data"]["mouse_v1"]["dataset_config"]["paths"])

            if "syn_data_percentage" in run:
                syn_data_percentage = np.round(run["syn_data_percentage"], 2)
            elif "syn_dataset_config" in run_config["data"] \
                and run_config["data"]["syn_dataset_config"] is not None \
                and run_config["data"]["syn_dataset_config"]["batch_size"] > 0:
                total_batch_size = base_data_batch_size + (run_config["data"]["syn_dataset_config"]["batch_size"] * len(run_config["data"]["syn_dataset_config"]["data_keys"]))
                syn_data_percentage = (run_config["data"]["syn_dataset_config"]["batch_size"] * len(run_config["data"]["syn_dataset_config"]["data_keys"])) / total_batch_size
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
            # ax[l_idx].plot(syn_data_percentages, l_vals, **plot_kwargs)
            sns.lineplot(x=syn_data_percentages, y=l_vals, ax=ax[l_idx], errorbar="se", **plot_kwargs)

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

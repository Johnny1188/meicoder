import os
import random
import numpy as np
import matplotlib.pyplot as plt
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

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import plot_losses, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, crop
from csng.losses import SSIMLoss, MSELossWithCrop, Loss, CroppedLoss
from csng.readins import MultiReadIn, FCReadIn, ConvReadIn

from cat_v1_spiking_model.dataset_50k.encoder import get_encoder
from cat_v1_spiking_model.dataset_50k.data import (
    prepare_v1_dataloaders,
    SyntheticDataset,
    BatchPatchesDataLoader,
    MixedBatchLoader,
    PerSampleStoredDataset,
)
from csng.readins import (
    MultiReadIn,
    HypernetReadIn,
    ConvReadIn,
    AttentionReadIn,
    FCReadIn,
    AutoEncoderReadIn,
    Conv1dReadIn,
    LocalizedFCReadIn,
    MEIReadIn,
)
from cat_v1_spiking_model.dataset_50k.multireadin_decoder_utils import train, val, get_dataloaders

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")



##### set run config #####
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
    },
    # "crop_win": (slice(15, 35), slice(15, 35)),
    "crop_win": (20, 20),
    "only_cat_v1_eval": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "wandb": None,
    "wandb": {
        "project": "CSNG",
        "group": "cat_v1_50k",
    },
}
config["data"]["cat_v1"] = {
    "train_path": os.path.join(DATA_PATH, "datasets", "train"),
    "val_path": os.path.join(DATA_PATH, "datasets", "val"),
    "test_path": os.path.join(DATA_PATH, "datasets", "test"),
    "image_size": [50, 50],
    "crop": False,
    "batch_size": 64,
    "stim_keys": ("stim",),
    "resp_keys": ("exc_resp", "inh_resp"),
    "return_coords": True,
    "return_ori": False,
    "coords_ori_filepath": os.path.join(DATA_PATH, "pos_and_ori.pkl"),
    "cached": False,
    "stim_normalize_mean": 46.143,
    "stim_normalize_std": 20.420,
    "resp_normalize_mean": torch.load(
        os.path.join(DATA_PATH, "responses_mean.pt")
    ),
    "resp_normalize_std": torch.load(
        os.path.join(DATA_PATH, "responses_std.pt")
    ),
}

config["decoder"] = {
    "model": {
        "readins_config": [
            {
                "data_key": "cat_v1",
                "in_shape": 46875,
                "decoding_objective_config": None,
                "layers": [
                    (ConvReadIn, {
                        "H": 10,
                        "W": 10,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 3, # x, y, resp
                            "layers_config": [("fc", 64), ("fc", 128), ("fc", 10*10)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": 46875,
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.1,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),

                    # (FCReadIn, {
                    #     "in_shape": 46875,
                    #     "layers_config": [
                    #         ("fc", 768),
                    #         ("unflatten", 1, (12, 8, 8)),
                    #     ],
                    #     "act_fn": nn.LeakyReLU,
                    #     "out_act_fn": nn.Identity,
                    #     "batch_norm": True,
                    #     "dropout": 0.15,
                    #     "out_channels": 12,
                    # }),

                    # (MEIReadIn, {
                    #     "meis_path": os.path.join(DATA_PATH, "meis", "cat_v1",  "meis.pt"),
                    #     "n_neurons": 46875,
                    #     "mei_resize_method": "crop",
                    #     "mei_target_shape": (20, 20),
                    #     "pointwise_conv_config": {
                    #         "out_channels": 480,
                    #         "bias": False,
                    #         "batch_norm": True,
                    #         "act_fn": nn.LeakyReLU,
                    #         "dropout": 0.15,
                    #     },
                    #     "ctx_net_config": {
                    #         "in_channels": 3, # resp, x, y
                    #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 20*20)],
                    #         "act_fn": nn.LeakyReLU,
                    #         "out_act_fn": nn.Identity,
                    #         "dropout": 0.15,
                    #         "batch_norm": True,
                    #     },
                    #     "shift_coords": False,
                    #     "device": config["device"],
                    # }),

                ],
            }
        ],
        "core_cls": CNN_Decoder,
        "core_config": {
            "resp_shape": [480],
            "stim_shape": [1, 50, 50],
            "layers": [
                ### Conv/FC readin
                ("deconv", 480, 7, 2, 3),
                ("deconv", 256, 5, 1, 2),
                ("deconv", 256, 5, 1, 2),
                ("deconv", 128, 4, 1, 1),
                ("deconv", 64, 3, 1, 1),
                ("deconv", 1, 3, 1, 0),

                ### MEI readin
                # ("conv", 480, 7, 1, 3),
                # ("conv", 256, 5, 1, 2),
                # ("conv", 256, 5, 1, 2),
                # ("conv", 128, 3, 1, 1),
                # ("conv", 64, 3, 1, 1),
                # ("conv", 1, 3, 1, 1),
            ],
            "act_fn": nn.ReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.35,
            "batch_norm": True,
        },
    },
    "opter_cls": torch.optim.AdamW,
    "opter_kwargs": {
        "lr": 3e-4,
        "weight_decay": 0.03,
    },
    "loss": {
        # "loss_fn": nn.MSELoss(),
        # "loss_fn": MSELossWithCrop(window=config["crop_win"]),
        "loss_fn": SSIMLoss(
            window=config["crop_win"],
            log_loss=True,
            inp_normalized=True,
            inp_standardized=False,
        ),
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
        "con_reg_mul": 0,
        # "con_reg_mul": 1,
        "con_reg_loss_fn": SSIMLoss(window=config["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        "encoder": None,
        # "encoder": get_encoder(
        #     device=config["device"],
        #     eval_mode=True,
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter.pth"),
        # ),
    },
    "n_epochs": 50,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_only_core": False,
    #     # "load_only_core": True,
    #     "ckpt_path": os.path.join(
    #         # DATA_PATH, "models", "cat_v1_pretraining", "2024-02-27_19-17-39", "decoder.pt"),
    #         DATA_PATH, "models", "cnn", "2024-04-23_11-56-13", "ckpt", "decoder_20.pt"),
    #         # DATA_PATH, "models", "cnn", "2024-03-27_10-39-16", "decoder.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "0rx79gwn"
    # },
    "save_run": True,
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### data
    dataloaders = dict()
    dataloaders["cat_v1"] = prepare_v1_dataloaders(**config["data"]["cat_v1"])

    ### sample data
    sample_data_key = "cat_v1"
    datapoint = next(iter(dataloaders["cat_v1"]["test"]))
    stim, resp, neuron_coords = datapoint.images, datapoint.responses, datapoint.neuron_coords.float().to(config["device"])

    ### decoder
    ### initialize (and load ckpt if needed)
    if config["decoder"]["load_ckpt"] != None:
        print(f"[INFO] Loading checkpoint from {config['decoder']['load_ckpt']['ckpt_path']}...")
        ckpt = torch.load(config["decoder"]["load_ckpt"]["ckpt_path"], map_location=config["device"], pickle_module=dill)

        if config["decoder"]["load_ckpt"]["load_only_core"]:
            print("[INFO] Loading only the core of the model (no history, no best ckpt)...")

            ### init decoder (load only the core)
            config["decoder"]["model"]["core_cls"] = ckpt["config"]["decoder"]["model"]["core_cls"]
            config["decoder"]["model"]["core_config"] = ckpt["config"]["decoder"]["model"]["core_config"]
            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict({k:v for k,v in ckpt["best"]["model"].items() if "readin" not in k}, strict=False)

            ### init the rest
            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
            history = {"train_loss": [], "val_loss": []}
            best = {"val_loss": np.inf, "epoch": 0, "model": None}
        else:
            print("[INFO] Continuing the training run (loading the current model, history, and overwriting the config)...")
            history, best, config["decoder"]["model"] = ckpt["history"], ckpt["best"], ckpt["config"]["decoder"]["model"]

            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict(ckpt["decoder"])

            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            opter.load_state_dict(ckpt["opter"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
    else:
        print("[INFO] Initializing the model from scratch...")
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
        
        history = {"train_loss": [], "val_loss": []}
        best = {"val_loss": np.inf, "epoch": 0, "model": None}

    ### print model and fix sizes of stimuli
    with torch.no_grad():
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords)
        print(stim_pred.shape)
        del stim_pred

    print(
        decoder,
        f"\n\n---"
        f"Number of parameters:"
        f"\n  whole model: {count_parameters(decoder)}"
        f"\n  core: {count_parameters(decoder.core)} ({count_parameters(decoder.core) / count_parameters(decoder) * 100:.2f}%)"
        f"\n  readins: {count_parameters(decoder.readins)} ({count_parameters(decoder.readins) / count_parameters(decoder) * 100:.2f}%)"
        f"\n    ({', '.join([f'{k}: {count_parameters(v)} [{count_parameters(v) / count_parameters(decoder) * 100:.2f}%]' for k, v in decoder.readins.items()])})"
    )

    ### prepare checkpointing and wandb logging
    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["decoder"]["save_run"]:
            ### save config
            config["dir"] = os.path.join(DATA_PATH, "models", "cnn", config["run_name"])
            os.makedirs(config["dir"], exist_ok=True)
            with open(os.path.join(config["dir"], "config.json"), "w") as f:
                json.dump(config, f, indent=4, default=str)
            os.makedirs(os.path.join(config["dir"], "samples"), exist_ok=True)
            os.makedirs(os.path.join(config["dir"], "ckpt"), exist_ok=True)
            make_sample_path = lambda epoch, prefix: os.path.join(
                config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
            )
            print(f"Run name: {config['run_name']}\nRun dir: {config['dir']}")
        else:
            make_sample_path = lambda epoch, prefix: None
            print("[WARNING] Not saving the run and the config.")
    else:
        config["run_name"] = ckpt["config"]["run_name"]
        config["dir"] = ckpt["config"]["dir"]
        make_sample_path = lambda epoch, prefix: os.path.join(
            config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
        )
        print(f"Checkpointing resumed - Run name: {config['run_name']}\nRun dir: {config['dir']}")

    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_wandb_id"] == None:
        if config["wandb"]:
            wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config,
                tags=[
                    config["decoder"]["model"]["core_cls"].__name__,
                    config["decoder"]["model"]["readins_config"][0]["layers"][0][0].__name__,
                ],
                notes=None)
            wdb_run.watch(decoder)
        else:
            print("[WARNING] Not using wandb.")
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must")

    ### train
    print(f"[INFO] Config:\n{json.dumps(config, indent=2, default=str)}")
    s, e = len(history["train_loss"]), config["decoder"]["n_epochs"]
    for epoch in range(s, e):
        print(f"[{epoch}/{e}]")

        ### train and val
        train_dataloader, val_dataloader, _ = get_dataloaders(
            config=config,
            dataloaders=dataloaders,
            only_cat_v1_eval=config["only_cat_v1_eval"],
        )
        train_loss = train(
            model=decoder,
            dataloader=train_dataloader,
            opter=opter,
            loss_fn=loss_fn,
            config=config,
        )
        val_loss = val(
            model=decoder,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            config=config,
        )

        ### save best model
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = epoch
            best["model"] = deepcopy(decoder.state_dict())

        ### log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"{train_loss=:.4f}, {val_loss=:.4f}")
        if config["wandb"]: wdb_run.log({"train_loss": train_loss, "val_loss": val_loss}, commit=False)

        ### plot reconstructions
        stim_pred = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[:8], data_key="cat_v1").detach()
        if "cat_v1" in config["data"] and config["data"]["cat_v1"]["crop"] == False:
            fig = plot_comparison(target=crop(stim[:8], config["crop_win"]).cpu(), pred=crop(stim_pred[:8], config["crop_win"]).cpu(), save_to=make_sample_path(epoch, ""), show=False)
        else:
            fig = plot_comparison(target=stim[:8].cpu(), pred=stim_pred[:8].cpu(), save_to=make_sample_path(epoch, "no_crop_"), show=False)
        if config["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(config["dir"], f"losses_{epoch}.png") if config["decoder"]["save_run"] else None)

        ### save ckpt
        if epoch % 5 == 0 and epoch > 0:
            ### ckpt
            if config["decoder"]["save_run"]:
                torch.save({
                    "decoder": decoder.state_dict(),
                    "opter": opter.state_dict(),
                    "history": history,
                    "config": config,
                    "best": best,
                }, os.path.join(config["dir"], "ckpt", f"decoder_{epoch}.pt"), pickle_module=dill)

    ### final evaluation + logging + saving
    print(f"Best val loss: {best['val_loss']:.4f} at epoch {best['epoch']}")

    ### save final ckpt
    if config["decoder"]["save_run"]:
        torch.save({
            "decoder": decoder.state_dict(),
            "opter": opter.state_dict(),
            "history": history,
            "config": config,
            "best": best,
        }, os.path.join(config["dir"], f"decoder.pt"), pickle_module=dill)

    ### eval on test set w/ current params
    print("Evaluating on test set with current model...")
    _, _, test_dataloader = get_dataloaders(
        config=config,
        dataloaders=dataloaders,
        only_cat_v1_eval=config["only_cat_v1_eval"],
    )
    curr_test_loss = val(
        model=decoder,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        config=config,
    )
    print(f"  Test loss (current model): {curr_test_loss:.4f}")

    ### load best model
    decoder.load_state_dict(best["model"])

    ### eval on test set w/ best params
    print("Evaluating on test set with the best model...")
    _, _, test_dataloader = get_dataloaders(
        config=config,
        dataloaders=dataloaders,
        only_cat_v1_eval=config["only_cat_v1_eval"],
    )
    final_test_loss = val(
        model=decoder,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        config=config,
    )
    print(f"  Test loss (best model): {final_test_loss:.4f}")

    ### plot reconstructions of the final model
    stim_pred_best = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[:8], data_key="cat_v1").detach().cpu()
    fig = plot_comparison(
        target=crop(stim[:8], config["crop_win"]).cpu(),
        pred=crop(stim_pred_best[:8], config["crop_win"]).cpu(),
        show=False,
        save_to=os.path.join(config["dir"], "stim_comparison_best.png") if config["decoder"]["save_run"] else None,
    )

    ### log
    if config["wandb"]:
        wandb.run.summary["best_val_loss"] = best["val_loss"]
        wandb.run.summary["best_epoch"] = best["epoch"]
        wandb.run.summary["curr_test_loss"] = curr_test_loss
        wandb.run.summary["final_test_loss"] = final_test_loss
        wandb.run.summary["best_reconstruction"] = fig

    ### save/delete wandb run
    if config["wandb"]:
        print("Finishing wandb run...")
        wdb_run.finish()

    ### plot losses
    plot_losses(
        history=history,
        show=False,
        save_to=None if not config["decoder"]["save_run"] else os.path.join(config["dir"], f"losses.png"),
    )

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
from csng.losses import SSIMLoss, Loss, CroppedLoss
from csng.readins import MultiReadIn, FCReadIn, ConvReadIn

from encoder import get_encoder
from data import (
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
from multireadin_decoder_all_data_utils import train, val, get_dataloaders

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")


##### set run config #####
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "device": os.environ["DEVICE"],
    "seed": 0,
    "wandb": None,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "cat_v1",
    },
}

### cat v1 data
config["data"]["cat_v1"] = {
    "crop_win": (20, 20),
    "dataset_config": {
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
        # "training_sample_idxs": np.random.choice(45000, size=22330, replace=False),
    },
}

### mouse v1 data
# config["data"]["mouse_v1"] = {
#     "dataset_fn": "sensorium.datasets.static_loaders",
#     "dataset_config": {
#         "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
#             # os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
#             # os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
#             os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
#             os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
#             os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
#             os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
#             os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
#         ],
#         "normalize": True,
#         "scale": 0.25, # 256x144 -> 64x36
#         "include_behavior": False,
#         "add_behavior_as_channels": False,
#         "include_eye_position": True,
#         "exclude": None,
#         "file_tree": True,
#         "cuda": "cuda" in config["device"],
#         "batch_size": 4,
#         "seed": config["seed"],
#         "use_cache": False,
#     },
#     "crop_win": (22, 36),
#     "skip_train": False,
#     "skip_val": False,
#     "skip_test": False,
#     "normalize_neuron_coords": True,
#     "average_test_multitrial": True,
#     "save_test_multitrial": True,
#     "test_batch_size": 7,
#     "device": config["device"],
# }

config["decoder"] = {
    "model": {
        "readins_config": [
            {
                "data_key": "cat_v1",
                "in_shape": 46875,
                "decoding_objective_config": None,
                "layers": [
                    # (ConvReadIn, {
                    #     "H": 10,
                    #     "W": 10,
                    #     "shift_coords": False,
                    #     "learn_grid": True,
                    #     "grid_l1_reg": 8e-3,
                    #     "in_channels_group_size": 1,
                    #     "grid_net_config": {
                    #         "in_channels": 3, # x, y, resp
                    #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 10*10)],
                    #         "act_fn": nn.LeakyReLU,
                    #         "out_act_fn": nn.Identity,
                    #         "dropout": 0.15,
                    #         "batch_norm": False,
                    #     },
                    #     "pointwise_conv_config": {
                    #         "in_channels": 46875,
                    #         "out_channels": 480,
                    #         "act_fn": nn.Identity,
                    #         "bias": False,
                    #         "batch_norm": True,
                    #         "dropout": 0.1,
                    #     },
                    #     "gauss_blur": False,
                    #     "gauss_blur_kernel_size": 7,
                    #     "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                    #     "gauss_blur_sigma_init": 1.5,
                    #     "neuron_emb_dim": None,
                    #     # "neuron_idxs": np.random.choice(46875, size=234, replace=False),
                    # }),

                    # (FCReadIn, {
                    #     "in_shape": 46875,
                    #     "layers_config": [
                    #         ("fc", 512),
                    #         ("unflatten", 1, (8, 8, 8)),
                    #     ],
                    #     "act_fn": nn.LeakyReLU,
                    #     "out_act_fn": nn.Identity,
                    #     "batch_norm": True,
                    #     "dropout": 0.15,
                    #     "out_channels": 8,
                    # }),

                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH, "meis", "cat_v1",  "meis.pt"),
                        "n_neurons": 46875,
                        "mei_resize_method": "resize",
                        "mei_target_shape": (20, 20),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.15,
                        },
                        "ctx_net_config": {
                            "in_channels": 3, # resp, x, y
                            "layers_config": [("fc", 64), ("fc", 128), ("fc", 20*20)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": True,
                        },
                        "shift_coords": False,
                        "device": config["device"],
                    }),
                ],
            }
        ] if "cat_v1" in config["data"] else [],
        "core_cls": CNN_Decoder,
        "core_config": {
            "resp_shape": (480,),
            "layers": [
                ### Conv/FC readin
                # ("deconv", 480, 7, 2, 2),
                # ("deconv", 256, 5, 1, 2),
                # ("deconv", 256, 5, 1, 2),
                # ("deconv", 128, 4, 1, 1),
                # ("deconv", 64, 3, 1, 1),
                # ("deconv", 1, 3, 1, 0),

                ### MEI readin
                ("conv", 480, 7, 1, 3),
                ("conv", 256, 5, 1, 2),
                ("conv", 256, 5, 1, 2),
                ("conv", 128, 3, 1, 1),
                ("conv", 64, 3, 1, 1),
                ("conv", 1, 3, 1, 1),
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
        "loss_fn": {
            "cat_v1": SSIMLoss(window=config["data"]["cat_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        },
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
        "con_reg_mul": 0,
        # "con_reg_mul": 1,
        "con_reg_loss_fn": {
            "cat_v1": SSIMLoss(window=config["data"]["cat_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        },
        "encoder": None,
        # "encoder": get_encoder(
        #     device=config["device"],
        #     eval_mode=True,
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter.pth"),
        # ),
    },
    "n_epochs": 60,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_only_core": False,
    #     "ckpt_path": os.path.join(
    #         # DATA_PATH, "models", "cat_v1_pretraining", "2024-02-27_19-17-39", "decoder.pt"),
    #         DATA_PATH, "models", "cnn", "2024-06-17_17-29-25", "ckpt", "decoder_40.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "znjuxlru",
    # },
    "save_run": True,
}

### append readins and losses for mouse v1
if "mouse_v1" in config["data"]:
    _dls, _neuron_coords = get_dataloaders(config=config)
    for data_key, n_coords in _dls["train"]["mouse_v1"].neuron_coords.items():
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_coords.shape[-2],
            "decoding_objective_config": None,
            "layers": [
                # (ConvReadIn, {
                #     "H": 10,
                #     "W": 18,
                #     "shift_coords": False,
                #     "learn_grid": True,
                #     "grid_l1_reg": 8e-3,
                #     "in_channels_group_size": 1,
                #     "grid_net_config": {
                #         "in_channels": 3, # x, y, resp
                #         "layers_config": [("fc", 32), ("fc", 86), ("fc", 18*10)],
                #         "act_fn": nn.LeakyReLU,
                #         "out_act_fn": nn.Identity,
                #         "dropout": 0.1,
                #         "batch_norm": False,
                #     },
                #     "pointwise_conv_config": {
                #         "in_channels": n_coords.shape[-2],
                #         "out_channels": 480,
                #         "act_fn": nn.Identity,
                #         "bias": False,
                #         "batch_norm": True,
                #         "dropout": 0.1,
                #     },
                #     "gauss_blur": False,
                #     "gauss_blur_kernel_size": 7,
                #     "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                #     "gauss_blur_sigma_init": 1.5,
                #     "neuron_emb_dim": None,
                # }),

                # (FCReadIn, {
                #     "in_shape": n_coords.shape[-2],
                #     "layers_config": [
                #         ("fc", 432),
                #         ("unflatten", 1, (3, 9, 16)),
                #     ],
                #     "act_fn": nn.LeakyReLU,
                #     "out_act_fn": nn.Identity,
                #     "batch_norm": True,
                #     "dropout": 0.15,
                # }),

                (MEIReadIn, {
                    "meis_path": os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "meis", data_key,  "meis.pt"),
                    "n_neurons": n_coords.shape[-2],
                    "mei_resize_method": "resize",
                    "mei_target_shape": (22, 36),
                    "pointwise_conv_config": {
                        "out_channels": 480,
                        "bias": False,
                        "batch_norm": True,
                        "act_fn": nn.LeakyReLU,
                        "dropout": 0.1,
                    },
                    "ctx_net_config": {
                        "in_channels": 3, # resp, x, y
                        "layers_config": [("fc", 32), ("fc", 128), ("fc", 22*36)],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "dropout": 0.1,
                        "batch_norm": True,
                    },
                    "shift_coords": False,
                    "device": config["device"],
                }),

            ],
        })
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["data"]["mouse_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)
        config["decoder"]["loss"]["con_reg_loss_fn"][data_key] = SSIMLoss(window=config["data"]["mouse_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)



if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### data
    dls, neuron_coords = get_dataloaders(config=config)

    ### sample data
    sample_dataset = "cat_v1"
    c_dp = next(iter(dls["val"][sample_dataset]))
    stim, resp, sample_data_key = c_dp[0]["stim"], c_dp[0]["resp"], c_dp[0]["data_key"]
    if "mouse_v1" in config["data"]:
        m_sample_dataset = "mouse_v1"
        m_dp = next(iter(dls["val"][m_sample_dataset]))
        m_stim, m_resp, m_sample_data_key, m_pupil_center = m_dp[0]["stim"], m_dp[0]["resp"], m_dp[0]["data_key"], m_dp[0]["pupil_center"]

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
            if "training_sample_idxs" in ckpt["config"]["data"]["cat_v1"]["dataset_config"]:
                config["data"]["cat_v1"]["dataset_config"]["training_sample_idxs"] = ckpt["config"]["data"]["cat_v1"]["dataset_config"]["training_sample_idxs"]

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
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords[sample_dataset])
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
        dls, neuron_coords = get_dataloaders(config=config)
        train_loss = train(
            model=decoder,
            dataloaders=dls["train"],
            opter=opter,
            loss_fn=loss_fn,
            config=config,
        )
        val_loss = val(
            model=decoder,
            dataloaders=dls["val"],
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
        stim_pred = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[sample_dataset], data_key=sample_data_key).detach()
        fig = plot_comparison(target=crop(stim[:8], config["data"][sample_dataset]["crop_win"]).cpu(), pred=crop(stim_pred[:8], config["data"][sample_dataset]["crop_win"]).cpu(), save_to=make_sample_path(epoch, "c_"), show=False)
        if "mouse_v1" in config["data"]:
            m_stim_pred = decoder(m_resp[:8].to(config["device"]), neuron_coords=neuron_coords[m_sample_dataset][m_sample_data_key], pupil_center=m_pupil_center[:8].to(config["device"]), data_key=m_sample_data_key).detach()
            fig = plot_comparison(target=crop(m_stim[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(), pred=crop(m_stim_pred[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(), save_to=make_sample_path(epoch, "m_"), show=False)
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
    dls, neuron_coords = get_dataloaders(config=config)
    curr_test_loss = val(
        model=decoder,
        dataloaders=dls["test"],
        loss_fn=loss_fn,
        config=config,
    )
    print(f"  Test loss (current model): {curr_test_loss:.4f}")

    ### load best model
    decoder.load_state_dict(best["model"])

    ### eval on test set w/ best params
    print("Evaluating on test set with the best model...")
    dls, neuron_coords = get_dataloaders(config=config)
    final_test_loss = val(
        model=decoder,
        dataloaders=dls["test"],
        loss_fn=loss_fn,
        config=config,
    )
    print(f"  Test loss (best model): {final_test_loss:.4f}")

    ### plot reconstructions of the final model
    stim_pred_best = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[sample_dataset], data_key=sample_data_key).detach().cpu()
    fig = plot_comparison(
        target=crop(stim[:8], config["data"][sample_dataset]["crop_win"]).cpu(),
        pred=crop(stim_pred_best[:8], config["data"][sample_dataset]["crop_win"]).cpu(),
        show=False,
        save_to=os.path.join(config["dir"], "c_stim_comparison_best.png") if config["decoder"]["save_run"] else None,
    )
    if "mouse_v1" in config["data"]:
        m_stim_pred_best = decoder(m_resp[:8].to(config["device"]), neuron_coords=neuron_coords[m_sample_dataset][m_sample_data_key], pupil_center=m_pupil_center[:8].to(config["device"]), data_key=m_sample_data_key).detach().cpu()
        fig = plot_comparison(
            target=crop(m_stim[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(),
            pred=crop(m_stim_pred_best[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(),
            show=False,
            save_to=os.path.join(config["dir"], "m_stim_comparison_best.png") if config["decoder"]["save_run"] else None,
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

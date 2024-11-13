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
from csng.GAN import GAN
from csng.utils import plot_losses, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, crop
from csng.losses import SSIMLoss, Loss, CroppedLoss
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

from csng.cat_v1.encoder import get_encoder
from csng.cat_v1.gan_all_data_utils import train, val, get_dataloaders

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")


##### set run config #####
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": 1489,
    },
    "device": os.environ["DEVICE"],
    "seed": 0,
    "crop_wins": dict(),
    "save_run": False,
    "wandb": None,
    "save_run": True,
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
        "batch_size": 15,
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
config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]

### mouse v1 data
config["data"]["mouse_v1"] = {
    "dataset_fn": "sensorium.datasets.static_loaders",
    "dataset_config": {
        "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
            # os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
            # os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
            os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
            os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
            os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
            os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
            os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22", "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
        ],
        "normalize": True,
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 3,
        "seed": config["seed"],
        "use_cache": False,
    },
    "crop_win": (22, 36),
    "skip_train": False,
    "skip_val": False,
    "skip_test": False,
    "normalize_neuron_coords": True,
    "average_test_multitrial": True,
    "save_test_multitrial": True,
    "test_batch_size": 7,
    "device": config["device"],
}

### synthetic mouse v1 data
config["data"]["syn_dataset_config"] = {
    "data_keys": [
        "21067-10-18",
        "22846-10-16",
        "23343-5-17",
        "23656-14-22",
        "23964-4-22",
    ],
    "batch_size": 3,
    "append_data_parts": ["train"],
    # "data_key_prefix": "syn",
    "data_key_prefix": None, # the same data key as the original (real) data
    "dir_name": "synthetic_data_mouse_v1_encoder_new_stimuli",
    "device": config["device"],
}


config["decoder"] = {
    "model": {
        "readins_config": [],
        # "readins_config": [
        #     {
        #         "data_key": "cat_v1",
        #         "in_shape": 46875,
        #         "decoding_objective_config": None,
        #         "layers": [
        #             # (ConvReadIn, {
        #             #     "H": 8,
        #             #     "W": 8,
        #             #     "shift_coords": False,
        #             #     "learn_grid": True,
        #             #     "grid_l1_reg": 8e-3,
        #             #     "in_channels_group_size": 1,
        #             #     "grid_net_config": {
        #             #         "in_channels": 3, # x, y, resp
        #             #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 8*8)],
        #             #         "act_fn": nn.LeakyReLU,
        #             #         "out_act_fn": nn.Identity,
        #             #         "dropout": 0.15,
        #             #         "batch_norm": False,
        #             #     },
        #             #     "pointwise_conv_config": {
        #             #         "in_channels": 46875,
        #             #         "out_channels": 480,
        #             #         "act_fn": nn.Identity,
        #             #         "bias": False,
        #             #         "batch_norm": True,
        #             #         "dropout": 0.1,
        #             #     },
        #             #     "gauss_blur": False,
        #             #     "gauss_blur_kernel_size": 7,
        #             #     "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
        #             #     # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
        #             #     "gauss_blur_sigma_init": 1.5,
        #             #     "neuron_emb_dim": None,
        #             # }),

        #             # (FCReadIn, {
        #             #     "in_shape": 46875,
        #             #     "layers_config": [
        #             #         ("fc", 512),
        #             #         ("unflatten", 1, (8, 8, 8)),
        #             #     ],
        #             #     "act_fn": nn.LeakyReLU,
        #             #     "out_act_fn": nn.Identity,
        #             #     "batch_norm": True,
        #             #     "dropout": 0.15,
        #             #     "out_channels": 8,
        #             # }),

        #             # (MEIReadIn, {
        #             #     "meis_path": os.path.join(DATA_PATH, "meis", "cat_v1",  "meis.pt"),
        #             #     "n_neurons": 46875,
        #             #     "mei_resize_method": "resize",
        #             #     "mei_target_shape": (20, 20),
        #             #     "pointwise_conv_config": {
        #             #         "out_channels": 480,
        #             #         "bias": False,
        #             #         "batch_norm": True,
        #             #         "act_fn": nn.LeakyReLU,
        #             #         "dropout": 0.15,
        #             #     },
        #             #     "ctx_net_config": {
        #             #         "in_channels": 3, # resp, x, y
        #             #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 20*20)],
        #             #         "act_fn": nn.LeakyReLU,
        #             #         "out_act_fn": nn.Identity,
        #             #         "dropout": 0.15,
        #             #         "batch_norm": True,
        #             #     },
        #             #     "shift_coords": False,
        #             #     "device": config["device"],
        #             # }),
        #         ],
        #     }
        # ] if "cat_v1" in config["data"] else [],
        "core_cls": GAN,
        "core_config": {
            "G_kwargs": {
                "in_shape": [480],
                "layers": [
                    # ("deconv", 480, 7, 2, 3),
                    # ("deconv", 256, 5, 1, 2),
                    # ("deconv", 256, 5, 1, 1),
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
            "D_kwargs": {
                "in_shape": [1, 20, 20],
                "layers": [
                    ("conv", 256, 7, 2, 2),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 128, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    dict(),
                ],
                "act_fn": nn.ReLU,
                "out_act_fn": nn.Identity, # sigmoid already in layers[-1] (head)
                "dropout": 0.3,
                "batch_norm": True,
            },
        },
    },
    "loss": {
        "loss_fn": dict(),
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
        "con_reg_mul": 0,
        # "con_reg_mul": 1,
        "con_reg_loss_fn": dict(),
        "encoder": None,
        # "encoder": get_encoder(
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall.pth"),
        #     device=config["device"],
        #     eval_mode=True,
        #     # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
        # ),
    },
    "val_loss": None,
    # "val_loss": get_metrics(config)["SSIML-PL"],
    "val_loss": "FID",
    "G_opter_cls": torch.optim.AdamW,
    "G_opter_kwargs": {"lr": 3e-4, "weight_decay": 0.03},
    "D_opter_cls": torch.optim.AdamW,
    "D_opter_kwargs": {"lr": 3e-4, "weight_decay": 0.03},
    "G_reg": {"l1": 0, "l2": 0},
    "D_reg": {"l1": 0, "l2": 0},
    "G_adv_loss_mul": 0.1,
    "G_stim_loss_mul": 0.9,
    "D_real_loss_mul": 0.5,
    "D_fake_loss_mul": 0.5,
    "D_real_stim_labels_noise": 0.05,
    "D_fake_stim_labels_noise": 0.05,
    "n_epochs": 50,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_best": False,
    #     "load_opter_state": True,
    #     "reset_history": False,
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-08-07_17-13-05", "ckpt", "decoder_48.pt"),
    #     # "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-24_09-36-46", "decoder.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "lpnyd1bz",
    # },
}

### append readins and losses for cat v1
if "cat_v1" in config["data"]:
    config["decoder"]["model"]["readins_config"].append({
        "data_key": "cat_v1",
        "in_shape": 46875,
        "decoding_objective_config": None,
        "layers": [
            # (ConvReadIn, {
            #     "H": 8,
            #     "W": 8,
            #     "shift_coords": False,
            #     "learn_grid": True,
            #     "grid_l1_reg": 8e-3,
            #     "in_channels_group_size": 1,
            #     "grid_net_config": {
            #         "in_channels": 3, # x, y, resp
            #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 8*8)],
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
            #     # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
            #     "gauss_blur_sigma_init": 1.5,
            #     "neuron_emb_dim": None,
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
    })
    config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1]["cat_v1"] = {
        "in_shape": [1, *config["data"]["cat_v1"]["crop_win"]],
        "layers_config": [("fc", 1)],
        "act_fn": nn.Identity,
        "out_act_fn": nn.Sigmoid,
    }
    config["decoder"]["loss"]["loss_fn"]["cat_v1"] = SSIMLoss(window=config["data"]["cat_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)
    config["decoder"]["loss"]["con_reg_loss_fn"]["cat_v1"] = SSIMLoss(window=config["data"]["cat_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)

### append readins and losses for mouse v1
if "mouse_v1" in config["data"]:
    _dls, _neuron_coords = get_dataloaders(config=config)
    for data_key, n_coords in _dls["train"]["mouse_v1"].neuron_coords.items():
        config["crop_wins"][data_key] = config["data"]["mouse_v1"]["crop_win"]
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
                #     # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
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
        config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1][data_key] = {
            "in_shape": [1, *config["data"]["mouse_v1"]["crop_win"]],
            "layers_config": [("fc", 1)],
            "act_fn": nn.Identity,
            "out_act_fn": nn.Sigmoid,
        }
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

        history = ckpt["history"]
        config["decoder"]["model"] = ckpt["config"]["decoder"]["model"]
        best = ckpt["best"]

        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        if config["decoder"]["load_ckpt"]["load_best"]:
            core_state_dict = {".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "G" in k or "D" in k}
        else:
            core_state_dict = {".".join(k.split(".")[1:]):v for k,v in ckpt["decoder"].items() if "G" in k or "D" in k}
        decoder.core.G.load_state_dict(core_state_dict["G"])
        decoder.core.D.load_state_dict(core_state_dict["D"])
        if config["decoder"]["load_ckpt"]["load_best"]:
            decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "readin" in k})
        else:
            decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in ckpt["decoder"].items() if "readin" in k})
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])
        if config["decoder"]["load_ckpt"]["load_opter_state"]:
            decoder.core.G_optim.load_state_dict(core_state_dict["G_optim"])
            decoder.core.D_optim.load_state_dict(core_state_dict["D_optim"])
        
        if config["decoder"]["load_ckpt"]["reset_history"]:
            history = {"val_loss": []}

        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
    else:
        print("[INFO] Initializing the model from scratch...")
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])
        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
        
        history = {"val_loss": []}
        best = {"val_loss": np.inf, "epoch": 0, "model": None}

    ### print model and fix sizes of stimuli
    with torch.no_grad():
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords[sample_dataset])
        print(stim_pred.shape)
        del stim_pred

    print(
        decoder,
        "\n---\n"
        f"Number of parameters:"
        f"\n\tTotal: {count_parameters(decoder)}"
        f"\n\tG: {count_parameters(decoder.core.G)}"
        f"\n\tD: {count_parameters(decoder.core.D)}"
    )

    ### prepare local checkpointing and tracking
    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["save_run"]:
            ### save config
            config["dir"] = os.path.join(DATA_PATH, "models", "gan", config["run_name"])
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

    ### prepare wandb logging
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
            wdb_run = None
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must")

    ### setup eval loss
    val_loss_fn = config["decoder"]["val_loss"]
    if val_loss_fn is None:
        val_loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])

    ### train
    print(f"[INFO] Config:\n{json.dumps(config, indent=2, default=str)}")
    s, e = len(history["val_loss"]), config["decoder"]["n_epochs"]
    for epoch in range(s, e):
        print(f"[{epoch}/{e}]")

        ### train and val
        dls, neuron_coords = get_dataloaders(config=config)
        history = train(
            model=decoder,
            dataloaders=dls["train"],
            loss_fn=loss_fn,
            config=config,
            history=history,
            wdb_run=wdb_run,
            wdb_commit=False,
        )
        val_loss = val(
            model=decoder,
            dataloaders=dls["val"],
            loss_fn=val_loss_fn,
            crop_wins=config["crop_wins"],
        )

        ### save best model
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = epoch
            best["model"] = deepcopy(decoder.state_dict())

        ### log
        history["val_loss"].append(val_loss)
        print(f"Validation loss={val_loss:.4f}")
        if config["wandb"]: wdb_run.log({"val_loss": val_loss}, commit=False)

        ### plot reconstructions
        stim_pred = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[sample_dataset], data_key=sample_data_key).detach()
        fig = plot_comparison(target=crop(stim[:8], config["data"][sample_dataset]["crop_win"]).cpu(), pred=crop(stim_pred[:8], config["data"][sample_dataset]["crop_win"]).cpu(), save_to=make_sample_path(epoch, "c_"), show=False)
        if "mouse_v1" in config["data"]:
            m_stim_pred = decoder(m_resp[:8].to(config["device"]), neuron_coords=neuron_coords[m_sample_dataset][m_sample_data_key], pupil_center=m_pupil_center[:8].to(config["device"]), data_key=m_sample_data_key).detach()
            fig = plot_comparison(target=crop(m_stim[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(), pred=crop(m_stim_pred[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(), save_to=make_sample_path(epoch, "m_"), show=False)
        if config["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(config["dir"], f"losses_{epoch}.png") if config["save_run"] else None)

        ### save ckpt
        if epoch % 3 == 0 and epoch > 0:
            ### ckpt
            if config["save_run"]:
                torch.save({
                    "decoder": decoder.state_dict(),
                    "history": history,
                    "config": config,
                    "best": best,
                }, os.path.join(config["dir"], "ckpt", f"decoder_{epoch}.pt"), pickle_module=dill)

    ### final evaluation + logging + saving
    print(f"Best val loss: {best['val_loss']:.4f} at epoch {best['epoch']}")

    ### save final ckpt
    if config["save_run"]:
        torch.save({
            "decoder": decoder.state_dict(),
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
        loss_fn=val_loss_fn,
        crop_wins=config["crop_wins"],
    )
    print(f"  Test loss (current model): {curr_test_loss:.4f}")

    ### load best model
    decoder.core.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "G" in k or "D" in k})
    decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "readin" in k})

    ### eval on test set w/ best params
    print("Evaluating on test set with the best model...")
    dls, neuron_coords = get_dataloaders(config=config)
    final_test_loss = val(
        model=decoder,
        dataloaders=dls["test"],
        loss_fn=val_loss_fn,
        crop_wins=config["crop_wins"],
    )
    print(f"  Test loss (best model): {final_test_loss:.4f}")

    ### plot reconstructions of the final model
    stim_pred_best = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[sample_dataset], data_key=sample_data_key).detach().cpu()
    fig = plot_comparison(
        target=crop(stim[:8], config["data"][sample_dataset]["crop_win"]).cpu(),
        pred=crop(stim_pred_best[:8], config["data"][sample_dataset]["crop_win"]).cpu(),
        show=False,
        save_to=os.path.join(config["dir"], "c_stim_comparison_best.png") if config["save_run"] else None,
    )
    if "mouse_v1" in config["data"]:
        m_stim_pred_best = decoder(m_resp[:8].to(config["device"]), neuron_coords=neuron_coords[m_sample_dataset][m_sample_data_key], pupil_center=m_pupil_center[:8].to(config["device"]), data_key=m_sample_data_key).detach().cpu()
        fig = plot_comparison(
            target=crop(m_stim[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(),
            pred=crop(m_stim_pred_best[:8], config["data"][m_sample_dataset]["crop_win"]).cpu(),
            show=False,
            save_to=os.path.join(config["dir"], "m_stim_comparison_best.png") if config["save_run"] else None,
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
        save_to=None if not config["save_run"] else os.path.join(config["dir"], f"losses.png"),
    )

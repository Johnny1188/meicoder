import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import lovely_tensors as lt
lt.monkey_patch()

from csng.models.gan import GAN
from csng.utils.mix import seed_all, plot_losses, plot_comparison, count_parameters, check_if_data_zscored
from csng.utils.data import standardize, normalize, crop
from csng.losses import SSIMLoss, MSELoss, Loss, get_metrics, VGGPerceptualLoss
from csng.models.readins import (
    MultiReadIn,
    ConvReadIn,
    FCReadIn,
    MEIReadIn,
)
from csng.data import get_dataloaders, get_sample_data
from csng.models.utils.gan import init_decoder, setup_run_dir, setup_wandb_run, train
from csng.utils.comparison import eval_decoder
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1
from csng.mouse_v1.encoder import get_encoder as get_encoder_mouse_v1

### set paths
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAE = os.path.join(os.environ["DATA_PATH"], "cae")
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")




##### global run config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    # "save_run": False,
    "save_run": True,
    # "wandb": None,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "gan_decoder",
    },
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_eval_batches": None,
        "eval_every_n_samples": 500,
    },
    "crop_wins": dict(),
}


### brainreader mouse data
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    # "batch_size": 2,
    # "batch_size": 5,
    "batch_size": 16,
    # "sessions": list(range(1, 23)),
    # "sessions": list(range(1, 9)),
    # "sessions": list(range(1, 7)),
    "sessions": [6],
    # "resize_stim_to": None,
    "resize_stim_to": (36, 64),
    # "resize_stim_to": (72, 128),
    # "resize_stim_to": (144, 256),
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
    "train_datapoint_idxs_to_use": None,
    # "train_datapoint_idxs_to_use": np.random.default_rng(seed=config["seed"]).choice(4500, size=int(4500 * 0.5), replace=False),
    # "train_datapoint_idxs_to_use": np.random.default_rng(seed=config["seed"]).choice(4500, size=2000, replace=False),
}
## add neuron coordinates to brainreader mouse data (learned by pretrained encoder)
# _enc_ckpt = torch.load(os.path.join(DATA_PATH, "models", "encoder_ball.pt"), pickle_module=dill)["model"]
# config["data"]["brainreader_mouse"]["neuron_coords"] = dict()
# for sess_id in config["data"]["brainreader_mouse"]["sessions"]:
#     config["data"]["brainreader_mouse"]["neuron_coords"][str(sess_id)] = _enc_ckpt[f"readout.{sess_id}._mu"][0,:,0].detach().clone()


### cat v1 data
# config["data"]["cat_v1"] = {
#     "crop_win": (20, 20),
#     "dataset_config": {
#         "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
#         "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
#         "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
#         "image_size": [50, 50],
#         "crop": False,
#         # "batch_size": 20,
#         "batch_size": 32,
#         "stim_keys": ("stim",),
#         "resp_keys": ("exc_resp", "inh_resp"),
#         "return_coords": True,
#         "return_ori": False,
#         "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
#         "cached": False,
#         "stim_normalize_mean": 46.143,
#         "stim_normalize_std": 24.960,
#         "resp_normalize_mean": None, # don't center responses
#         "resp_normalize_std": torch.load(
#             os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
#         ),
#         "clamp_neg_resp": False,
#         # "training_sample_idxs": np.random.choice(45000, size=5000, replace=False),
#     },
#     "neuron_coords_to_use": None, # if None, uses the neuron coordinates from the dataset (if return_coords=True)
# }
# ### use the neuron coordinates learned by pretrained encoder
# config["data"]["cat_v1"]["neuron_coords_to_use"] = get_encoder_cat_v1(
#     ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt")
# ).readout["cat_v1"].sample_grid(batch_size=1, sample=False)[0,:,0].detach().clone()

### mouse v1 data
# config["data"]["mouse_v1"] = {
#     "dataset_fn": "sensorium.datasets.static_loaders",
#     "dataset_config": {
#         "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
#             os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-1
#             # os.path.join(DATA_PATH_MOUSE_V1, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-2
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-3
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-4
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-5
#         ],
#         "normalize": True,
#         "z_score_responses": False,
#         "scale": 0.25, # 256x144 -> 64x36
#         "include_behavior": False,
#         "add_behavior_as_channels": False,
#         "include_eye_position": True,
#         "exclude": None,
#         "file_tree": True,
#         "cuda": "cuda" in config["device"],
#         # "batch_size": 2,
#         # "batch_size": 8,
#         "batch_size": 32,
#         "drop_last": True,
#         "use_cache": False,
#         "train_datapoint_idxs_to_use": None,
#         # "train_datapoint_idxs_to_use": np.random.default_rng(seed=config["seed"]).choice(4473, size=int(4473 * 0.5), replace=False),
#         # "train_datapoint_idxs_to_use": np.random.default_rng(seed=config["seed"]).choice(4473, size=100, replace=False),
#     },
#     "crop_win": (22, 36),
#     "skip_train": False,
#     "skip_val": False,
#     "skip_test": False,
#     "normalize_neuron_coords": True,
#     "average_test_multitrial": True,
#     "save_test_multitrial": True,
#     "test_batch_size": 7,
#     "neuron_coords_to_use": None, # if None, uses the neuron coordinates from the dataset
#     "device": config["device"],
# }
# use the neuron coordinates learned by pretrained encoder
# config["data"]["mouse_v1"]["neuron_coords_to_use"] = get_encoder_mouse_v1(
#     ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt")
# ).readout["21067-10-18"].sample_grid(batch_size=1, sample=False)[0,:,0]

# ### synthetic data
# config["data"]["syn_data"] = {
#     "data_dicts": [
#         {
#             "path": os.path.join(DATA_PATH, f"synthetic_data_{syn_data_key}_train", syn_data_key),
#             # "data_key": f"syn_{syn_data_key}",
#             "data_key": f"{syn_data_key}",
#             "load_neuron_coords": False,
#             "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis", syn_data_key,  "meis.pt"),
#         } for syn_data_key in ["6"]
#         # } for syn_data_key in [str(i) for i in range(1, 23)]
#     ],
#     "append_data_tiers": ["train"],
#     "responses_shift_mean": True,
#     "responses_clip_min": 0,
#     "responses_clip_max": None,

#     "device": config["device"],
#     "batch_size": 16,
#     # "batch_size": 2,
#     "shuffle": True,
#     "mixing_strategy": config["data"]["mixing_strategy"],
#     "max_training_batches": None,
#     # "max_training_batches": 282,
#     # "max_training_batches": 2250,
#     "return_pupil_center": False,
#     "return_neuron_coords": False,
#     "crop_win": (36, 64),
# }


### decoder
inp_zscored = check_if_data_zscored(cfg=config)
config["decoder"] = {
    "readin_type": (readin_type := "mei"), # "conv", "fc", "mei"
    "model": {
        "readins_config": [],
        "core_cls": GAN,
        "core_config": {
            "G_kwargs": {
                "in_shape": (480,), # needs to match the # of channels of the readin output
                "layers": {
                    "conv": [
                        ("deconv", 480, 7, 2, 3),
                        ("deconv", 256, 5, 1, 2),
                        ("deconv", 256, 5, 1, 1),
                        ("deconv", 128, 4, 1, 1),
                        ("deconv", 64, 3, 1, 1),
                        ("deconv", 1, 3, 1, 0),
                    ],
                    "fc": [
                        ("deconv", 480, 7, 2, 3),
                        ("deconv", 256, 5, 1, 2),
                        ("deconv", 256, 5, 1, 1),
                        ("deconv", 128, 4, 1, 1),
                        ("deconv", 64, 3, 1, 1),
                        ("deconv", 1, 3, 1, 0),
                    ],
                    "mei": [ # 36x64
                        ("conv", 480, 7, 1, 3),
                        ("conv", 256, 5, 1, 2),
                        ("conv", 256, 5, 1, 2),
                        ("conv", 128, 3, 1, 1),
                        ("conv", 64, 3, 1, 1),
                        ("conv", 1, 3, 1, 1),
                    ],
                    # "mei": [
                    #     ("deconv", 480, 7, 2, 3),
                    #     ("deconv", 256, 5, 1, 2),
                    #     ("deconv", 256, 5, 1, 1),
                    #     ("deconv", 128, 4, 1, 1),
                    #     ("deconv", 128, 3, 1, 1),
                    #     ("deconv", 64, 3, 1, 1),
                    #     ("deconv", 1, 3, 1, 1),
                    # ],
                    # "mei": [ # 72x128
                    #     ("deconv", 480, 7, 2, 3),
                    #     ("deconv", 256, 5, 1, 2),
                    #     ("deconv", 256, 5, 1, 1),
                    #     ("deconv", 128, 4, 1, 1),
                    #     ("deconv", 128, 3, 1, 1),
                    #     ("deconv", 64, 3, 1, 1),
                    #     ("deconv", 64, 3, 1, 1),
                    #     ("deconv", 1, 3, 1, 1),
                    # ],
                }[readin_type],
                "act_fn": nn.ReLU,
                "out_act_fn": nn.Identity,
                "dropout": 0.35,
                "batch_norm": True,
            },
            "D_kwargs": {
                "in_shape": [1, 36, 64],
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
        # "brain_distance_mul": 0.1,
        # "brain_distance_config": { # TODO: only works with brainreader data
        #     "encoder": get_encoder_brainreader(
        #         ckpt_path=os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
        #         eval_mode=True,
        #         device=config["device"],
        #     ),
        #     "use_gt_resp": True,
        #     "resp_loss_fn": F.mse_loss,
        #     "zscore_inp": False,
        #     "minmax_normalize_inp": False,
        # },
    },
    "eval_loss_name": "Alex(5) Loss",  # for "higher is better" metrics, use "<name> Loss"
    "G_opter_cls": torch.optim.AdamW,
    # "G_opter_kwargs": {"lr": 1e-4, "weight_decay": 0.08},
    "G_opter_kwargs": {"lr": 3e-5, "weight_decay": 0.3},
    # "G_opter_kwargs": {"lr": 3e-5, "weight_decay": 0.005},
    "D_opter_cls": torch.optim.AdamW,
    # "D_opter_kwargs": {"lr": 1e-4, "weight_decay": 0.08},
    "D_opter_kwargs": {"lr": 3e-5, "weight_decay": 0.3},
    # "D_opter_kwargs": {"lr": 3e-5, "weight_decay": 0.005},
    "G_reg": {"l1": 0, "l2": 0},
    "D_reg": {"l1": 0, "l2": 0},
    "G_adv_loss_mul": 0.1,
    "G_stim_loss_mul": 0.9,
    "D_real_loss_mul": 0.5,
    "D_fake_loss_mul": 0.5,
    "D_real_stim_labels_noise": 0.05,
    "D_fake_stim_labels_noise": 0.05,
    "n_epochs": 300,
    "load_ckpt": None,

    ### continue training
    # "load_ckpt": {
    #     "load_only_core": False,
    #     "load_best": False,
    #     "load_opter_state": True,
    #     "load_history": True,
    #     "reset_best": False,
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-10-02_11-09-09", "decoder.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "2025-10-02_11-09-09",
    # },
    ### for fine-tuning
    # "load_ckpt": {
    #     "load_only_core": True,
    #     "load_best": True,
    #     "load_opter_state": False,
    #     "load_history": False,
    #     "reset_best": True,
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-07_01-29-21", "ckpt", "decoder_200.pt"),
    #     "resume_checkpointing": False,
    #     "resume_wandb_id": None,
    # },
}

### finish config for brainreader mouse
if "brainreader_mouse" in config["data"]:
    _dls, _ = get_dataloaders(config=config)
    for data_key, dset in zip(_dls["train"]["brainreader_mouse"].data_keys, _dls["train"]["brainreader_mouse"].datasets):
        ### set crop wins and losses
        config["crop_wins"][data_key] = tuple(dset[0].images.shape[-2:])
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["crop_wins"][data_key], log_loss=True, inp_normalized=True, inp_standardized=False)
        # config["decoder"]["loss"]["loss_fn"][data_key] = Loss(config=dict(
        #     loss_fn=VGGPerceptualLoss(
        #         resize=False,
        #         device=config["device"],
        #         reduction="mean",
        #     ),
        #     window=config["crop_wins"][data_key],
        #     standardize=True,
        # ))

        ### append discriminator's head
        config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1][data_key] = {
            "in_shape": [1, 36, 64],
            # "in_shape": [1, 72, 128],
            # "in_shape": [1, 144, 256],
            "layers_config": [("fc", 1)],
            "act_fn": nn.Identity,
            "out_act_fn": nn.Sigmoid,
        }

        ### append readin
        n_neurons = dset[0].responses.shape[-1]
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_neurons,
            "decoding_objective_config": None,
            "layers": {
                "conv": [
                    (ConvReadIn, {
                        "H": 18,
                        "W": 32,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 1, # resp
                            "layers_config": [("fc", 8), ("fc", 64), ("fc", 18*32)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.2,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": n_neurons,
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.2,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),
                ],
                "fc": [
                    (FCReadIn, {
                        "in_shape": n_neurons,
                        "layers_config": [
                            ("fc", 1728),
                            ("unflatten", 1, (3, 18, 32)),
                        ],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "batch_norm": True,
                        "dropout": 0.15,
                        "apply_resp_transform": False,
                    }),
                ],
                "mei": [
                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis", data_key,  "meis.pt"),
                        # "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis_smallerstd_gauss2_encoder_b6_72-128", data_key,  "meis.pt"),
                        "n_neurons": n_neurons,
                        "mei_resize_method": "resize",
                        "mei_target_shape": (36, 64),
                        # "mei_target_shape": (72, 128),
                        # "mei_target_shape": (144, 256),
                        "meis_trainable": False,
                        "use_neuron_coords": (_use_neuron_coords := False),
                        "neuron_emb_dim": (_neuron_emb_dim := 32),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.2,
                        },
                        "ctx_net_config": {
                            "in_channels": 1 + 2*int(_use_neuron_coords) + (_neuron_emb_dim or 0), # resp, x, y, neuron_emb
                            "layers_config": [("fc", 128), ("fc", 36*64)],
                            # "layers_config": [("fc", 128), ("fc", 72*128)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": True,
                        },
                        "l2_reg_mul": 0,
                        "apply_resp_transform": False,
                        "shift_coords": False,
                        "neuron_idxs": None, # np.random.default_rng(seed=config["seed"]).choice(n_neurons, size=int(n_neurons * 0.5), replace=False),
                        # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(n_neurons, size=int(n_neurons * 0.05), replace=False),
                        # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(n_neurons, size=500, replace=False),
                        "device": config["device"],
                    }),
                ],
            }[config["decoder"]["readin_type"]],
        })

### finish config for cat v1
if "cat_v1" in config["data"]:
    ### set crop wins and losses
    config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]
    config["decoder"]["loss"]["loss_fn"]["cat_v1"] = SSIMLoss(window=config["crop_wins"]["cat_v1"], log_loss=True, inp_normalized=True, inp_standardized=False)
    # config["decoder"]["loss"]["loss_fn"]["cat_v1"] = Loss(config=dict(
    #     loss_fn=VGGPerceptualLoss(
    #         resize=False,
    #         device=config["device"],
    #         reduction="mean",
    #     ),
    #     window=config["crop_wins"]["cat_v1"],
    #     standardize=True,
    # ))

    ### append discriminator's head
    config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1]["cat_v1"] = {
        "in_shape": [1, *config["crop_wins"]["cat_v1"]],
        "layers_config": [("fc", 1)],
        "act_fn": nn.Identity,
        "out_act_fn": nn.Sigmoid,
    }

    ### append readin
    config["decoder"]["model"]["readins_config"].append({
        "data_key": "cat_v1",
        "in_shape": 46875,
        "decoding_objective_config": None,
        "layers": {
            "conv": [
                (ConvReadIn, {
                    "H": 8,
                    "W": 8,
                    "shift_coords": False,
                    "learn_grid": True,
                    "grid_l1_reg": 8e-3,
                    "in_channels_group_size": 1,
                    "grid_net_config": {
                        "in_channels": 3, # x, y, resp
                        "layers_config": [("fc", 64), ("fc", 128), ("fc", 8*8)],
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
            ],
            "fc": [
                (FCReadIn, {
                    "in_shape": 46875,
                    "layers_config": [
                        ("fc", 192),
                        ("unflatten", 1, (3, 8, 8)),
                    ],
                    "act_fn": nn.LeakyReLU,
                    "out_act_fn": nn.Identity,
                    "batch_norm": True,
                    "dropout": 0.15,
                    "out_channels": 8,
                }),
            ],
            "mei": [
                (MEIReadIn, {
                    "meis_path": os.path.join(DATA_PATH_CAT_V1, "meis", "cat_v1",  "meis.pt"),
                    "n_neurons": 46875,
                    "mei_resize_method": "resize",
                    "mei_target_shape": config["crop_wins"]["cat_v1"],
                    "meis_trainable": False,
                    "use_neuron_coords": (_use_neuron_coords := False),
                    "neuron_emb_dim": (_neuron_emb_dim := 16),
                    "pointwise_conv_config": {
                        "out_channels": 480,
                        "bias": False,
                        "batch_norm": True,
                        "act_fn": nn.LeakyReLU,
                        "dropout": 0.15,
                    },
                    "ctx_net_config": {
                        "in_channels": 1 + 2*int(_use_neuron_coords) + (_neuron_emb_dim or 0), # resp, x, y, neuron_emb
                        "layers_config": [("fc", 32), ("fc", 128), ("fc", np.prod(config["crop_wins"]["cat_v1"]))],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "dropout": 0.15,
                        "batch_norm": True,
                    },
                    "l2_reg_mul": 0,
                    "apply_resp_transform": False,
                    "shift_coords": False,
                    "neuron_idxs": None,
                    # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(46875, size=int(46875 * 0.005), replace=False),
                    # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(46875, size=500, replace=False),
                    "device": config["device"],
                }),
            ],
        }[config["decoder"]["readin_type"]],
    })

### finish config for mouse v1
if "mouse_v1" in config["data"]:
    _dls, _neuron_coords = get_dataloaders(config=config)
    for data_key, n_coords in _dls["train"]["mouse_v1"].neuron_coords.items():
        ### set crop wins and losses
        config["crop_wins"][data_key] = config["data"]["mouse_v1"]["crop_win"]
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["crop_wins"][data_key], log_loss=True, inp_normalized=True, inp_standardized=False)
        # config["decoder"]["loss"]["loss_fn"][data_key] = Loss(config=dict(
        #     loss_fn=VGGPerceptualLoss(
        #         resize=False,
        #         device=config["device"],
        #         reduction="mean",
        #     ),
        #     window=config["crop_wins"][data_key],
        #     standardize=True,
        # ))

        ### append discriminator's head
        config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1][data_key] = {
            "in_shape": [1, *config["crop_wins"][data_key]],
            "layers_config": [("fc", 1)],
            "act_fn": nn.Identity,
            "out_act_fn": nn.Sigmoid,
        }

        ### append readin
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_coords.shape[-2],
            "decoding_objective_config": None,
            "layers": {
                "conv": [
                    (ConvReadIn, {
                        "H": 10,
                        "W": 18,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 3, # x, y, resp
                            "layers_config": [("fc", 32), ("fc", 86), ("fc", 18*10)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.1,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": n_coords.shape[-2],
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.1,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),
                ],
                "fc": [
                    (FCReadIn, {
                        "in_shape": n_coords.shape[-2],
                        "layers_config": [
                            ("fc", 540),
                            ("unflatten", 1, (3, 10, 18)),
                        ],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "batch_norm": True,
                        "dropout": 0.15,
                    }),
                ],
                "mei": [
                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH_MOUSE_V1, "meis", data_key,  "meis.pt"),
                        "n_neurons": n_coords.shape[-2],
                        "mei_resize_method": "resize",
                        "mei_target_shape": config["crop_wins"][data_key],
                        "meis_trainable": False,
                        "use_neuron_coords": (_use_neuron_coords := False),
                        "neuron_emb_dim": (_neuron_emb_dim := 32),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.2,
                        },
                        "ctx_net_config": {
                            "in_channels": 1 + 2*int(_use_neuron_coords) + (_neuron_emb_dim or 0), # resp, x, y, neuron_emb
                            "layers_config": [("fc", 128), ("fc", np.prod(config["crop_wins"][data_key]))],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": True,
                        },
                        "l2_reg_mul": 0,
                        "l1_reg_mul": 0,
                        "apply_resp_transform": False,
                        "shift_coords": False,
                        "neuron_idxs": None,
                        # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(n_coords.shape[-2], size=int(n_coords.shape[-2] * 0.015), replace=False),
                        # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(n_coords.shape[-2], size=500, replace=False),
                        "device": config["device"],
                    }),
                ],
            }[config["decoder"]["readin_type"]],
        })

### finish config for synthetic data
if "syn_data" in config["data"]:
    _dls, _ = get_dataloaders(config=config)
    for data_key, syn_data_dict, dset in zip(
        _dls["train"]["syn_data"].data_keys,
        config["data"]["syn_data"]["data_dicts"],
        _dls["train"]["syn_data"].datasets,
    ):
        assert syn_data_dict["data_key"] == data_key, f"{syn_data_dict['data_key']} != {data_key}"

        ### set crop wins and losses
        config["crop_wins"][data_key] = tuple(dset[0].images.shape[-2:])
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["crop_wins"][data_key], log_loss=True, inp_normalized=True, inp_standardized=False)

        ### append discriminator's head
        config["decoder"]["model"]["core_config"]["D_kwargs"]["layers"][-1][data_key] = {
            "in_shape": [1, *config["crop_wins"][data_key]],
            "layers_config": [("fc", 1)],
            "act_fn": nn.Identity,
            "out_act_fn": nn.Sigmoid,
        }

        ### append readin
        n_neurons = dset[0].responses.shape[-1]
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_neurons,
            "decoding_objective_config": None,
            "layers": {
                "conv": [
                    (ConvReadIn, {
                        "H": 18,
                        "W": 32,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 1, # resp
                            "layers_config": [("fc", 8), ("fc", 64), ("fc", 18*32)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.2,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": n_neurons,
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.2,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),
                ],
                "fc": [
                    (FCReadIn, {
                        "in_shape": n_neurons,
                        "layers_config": [
                            ("fc", 1728),
                            ("unflatten", 1, (3, 18, 32)),
                        ],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "batch_norm": True,
                        "dropout": 0.15,
                    }),
                ],
                "mei": [
                    (MEIReadIn, {
                        "meis_path": syn_data_dict["meis_path"],
                        "n_neurons": n_neurons,
                        "mei_resize_method": "resize",
                        "mei_target_shape": (36, 64),
                        "meis_trainable": False,
                        "use_neuron_coords": (_use_neuron_coords := False),
                        "neuron_emb_dim": (_neuron_emb_dim := None),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.15,
                        },
                        "ctx_net_config": {
                            "in_channels": 1 + 2*int(_use_neuron_coords) + (_neuron_emb_dim or 0), # resp, x, y, neuron_emb
                            "layers_config": [("fc", 8), ("fc", 128), ("fc", 36*64)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": True,
                        },
                        "apply_resp_transform": False,
                        "shift_coords": False,
                        "neuron_idxs": None, # np.random.default_rng(seed=config["seed"]).choice(n_neurons, size=int(n_neurons * 0.5), replace=False),
                        "device": config["device"],
                    }),
                ],
            }[config["decoder"]["readin_type"]],
        })


### main pipeline
def run(cfg):
    print(f"... Running on {cfg['device']} ...")
    print(f"{DATA_PATH=}")
    seed_all(cfg["seed"])

    ### get data sample for plotting and logging
    dls, neuron_coords = get_dataloaders(config=cfg)
    s = get_sample_data(dls=dls, config=cfg, sample_from_tier="val")
    resp, sample_dataset, sample_data_key = s["resp"], s["sample_dataset"], s["sample_data_key"]

    ### init decoder (and load ckpt if needed)
    seed_all(cfg["seed"])
    cfg, decoder, loss_fn, history, best, ckpt = init_decoder(config=cfg)
    with torch.no_grad():
        print(
            decoder,
            f"\n\n-----"
            f"\nOutput shape: {decoder(resp.to(cfg['device']), data_key=sample_data_key, neuron_coords=(neuron_coords[sample_dataset][sample_data_key] if neuron_coords[sample_dataset] is not None else None)).shape}"
            f"\n-----"
            f"\nNumber of parameters:"
            f"\n  Total: {count_parameters(decoder)}"
            f"\n  G: {count_parameters(decoder.core.G)}"
            f"\n  D: {count_parameters(decoder.core.D)}"
        )

    ### prepare run name and run directory
    cfg, make_sample_path = setup_run_dir(config=cfg, ckpt=ckpt)

    ### prepare wandb logging
    wdb_run = setup_wandb_run(config=cfg, decoder=decoder)

    ### setup (e)val metric
    inp_zscored = check_if_data_zscored(cfg=cfg)
    val_metrics = {data_key: {
        cfg["decoder"]["eval_loss_name"]: get_metrics(
            inp_zscored=inp_zscored,
            crop_win=cfg["crop_wins"][data_key],
            device=cfg["device"])[cfg["decoder"]["eval_loss_name"]]
        } for data_key in cfg["crop_wins"].keys()
    }

    ### train
    print(f"[INFO] cfg:\n{json.dumps(cfg, indent=2, default=str)}")
    start, end = len(history["val_loss"]), cfg["decoder"]["n_epochs"]
    seed_all(cfg["seed"])
    for epoch in range(start, end):
        print(f"[{epoch}/{end}]")

        ### train and val
        dls, neuron_coords = get_dataloaders(config=cfg)
        history = train(
            model=decoder,
            dataloaders=dls["train"],
            loss_fn=loss_fn,
            config=cfg,
            history=history,
            wdb_run=wdb_run,
            wdb_commit=False,
        )
        val_loss = eval_decoder(
            model=decoder,
            dataloaders=dls["val"],
            loss_fns=val_metrics,
            crop_wins=cfg["crop_wins"],
            max_batches=cfg["data"]["max_eval_batches"],
            eval_every_n_samples=cfg["data"]["eval_every_n_samples"],
        )["total"][cfg["decoder"]["eval_loss_name"]]

        ### save best model
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = epoch
            best["decoder"] = deepcopy(decoder.state_dict())

        ### log
        print(f"Validation loss={val_loss:.4f}")
        if cfg["wandb"]: wdb_run.log({"val_loss": val_loss}, commit=False)
        history["val_loss"].append(val_loss)

        ### plot reconstructions
        if "brainreader_mouse" in cfg["data"]:
            b_stim_pred = decoder(s["b_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["b_stim"][:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(), pred=crop(b_stim_pred[:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "b_"), show=False)
        if "cat_v1" in cfg["data"]:
            c_stim_pred = decoder(s["c_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]][s["c_sample_data_key"]], data_key=s["c_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["c_stim"][:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(), pred=crop(c_stim_pred[:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "c_"), show=False)
        if "mouse_v1" in cfg["data"]:
            m_stim_pred = decoder(s["m_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"][:8].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["m_stim"][:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(), pred=crop(m_stim_pred[:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "m_"), show=False)
        if cfg["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(cfg["dir"], f"losses_{epoch}.png") if cfg["save_run"] else None)

        ### save ckpt
        if epoch % 5 == 0 and epoch > 0 and cfg["save_run"]:
            torch.save({
                "decoder": decoder.state_dict(),
                "history": history,
                "config": cfg,
                "best": best,
            }, os.path.join(cfg["dir"], "ckpt", f"decoder_{epoch}.pt"), pickle_module=dill)

    ### final evaluation + logging + saving
    seed_all(cfg["seed"])
    decoder.eval()
    print("\n" + "-" * 10 + f"\nBest val {cfg['decoder']['eval_loss_name']}: {best['val_loss']:.4f} at epoch {best['epoch']}")

    ### save final ckpt
    if cfg["save_run"]:
        torch.save({
            "decoder": decoder.state_dict(),
            "history": history,
            "config": cfg,
            "best": best,
        }, os.path.join(cfg["dir"], f"decoder.pt"), pickle_module=dill)

    ### eval on test set w/ current params
    print("Evaluating on test set with current model...")
    dls, neuron_coords = get_dataloaders(config=cfg)
    curr_test_loss = eval_decoder(
        model=decoder,
        dataloaders=dls["test"],
        loss_fns=val_metrics,
        crop_wins=cfg["crop_wins"],
    )["total"][cfg["decoder"]["eval_loss_name"]]
    print(f"  Test {cfg['decoder']['eval_loss_name']} (current model): {curr_test_loss:.4f}")

    ### load best model
    decoder.load_from_ckpt(ckpt={"best": best}, load_best=True, load_only_core=False, strict=True)

    ### eval on test set w/ best params
    print("Evaluating on test set with the best model...")
    dls, neuron_coords = get_dataloaders(config=cfg)
    final_test_loss = eval_decoder(
        model=decoder,
        dataloaders=dls["test"],
        loss_fns=val_metrics,
        crop_wins=cfg["crop_wins"],
    )["total"][cfg["decoder"]["eval_loss_name"]]
    print(f"  Test {cfg['decoder']['eval_loss_name']} (best model): {final_test_loss:.4f}")

    ### plot reconstructions of the final model
    if "brainreader_mouse" in cfg["data"]:
        b_stim_pred_best = decoder(s["b_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["b_stim"][:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(),
            pred=crop(b_stim_pred_best[:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "b_stim_comparison_best.png") if cfg["save_run"] else None,
        )
    if "cat_v1" in cfg["data"]:
        c_stim_pred_best = decoder(s["c_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]][s["c_sample_data_key"]], data_key=s["c_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["c_stim"][:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(),
            pred=crop(c_stim_pred_best[:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "c_stim_comparison_best.png") if cfg["save_run"] else None,
        )
    if "mouse_v1" in cfg["data"]:
        m_stim_pred_best = decoder(s["m_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"][:8].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["m_stim"][:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(),
            pred=crop(m_stim_pred_best[:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "m_stim_comparison_best.png") if cfg["save_run"] else None,
        )

    ### finish wandb run
    if cfg["wandb"]:
        wandb.run.summary["best_val_loss"] = best["val_loss"]
        wandb.run.summary["best_epoch"] = best["epoch"]
        wandb.run.summary["curr_test_loss"] = curr_test_loss
        wandb.run.summary["final_test_loss"] = final_test_loss
        wandb.run.summary["best_reconstruction"] = fig
        wdb_run.finish()

    ### plot losses
    plot_losses(
        history=history,
        show=False,
        save_to=None if not cfg["save_run"] else os.path.join(cfg["dir"], f"losses.png"),
    )


if __name__ == "__main__":
    run(cfg=config)

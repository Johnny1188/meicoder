import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.utils import crop, plot_comparison, standardize, normalize, count_parameters, plot_losses, slugify
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
from csng.comparison import load_decoder_from_ckpt, get_metrics, plot_reconstructions, plot_metrics, plot_syn_data_loss_curve

from encoder import get_encoder
from data_utils import get_mouse_v1_data
from comparison_utils import (
    find_best_ckpt,
    eval_decoder,
    get_all_data,
    plot_over_ckpts,
)

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


### global config
config = {
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
        "mouse_v1": None,
        "syn_dataset_config": None,
        "data_augmentation": None,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "crop_win": (22, 36),
    "wandb": None,
}

### prep data config
config["data"]["mouse_v1"] = {
    "dataset_fn": "sensorium.datasets.static_loaders",
    "dataset_config": {
        "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
            # os.path.join(DATA_PATH, "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
            # os.path.join(DATA_PATH, "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
            os.path.join(DATA_PATH, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
            # os.path.join(DATA_PATH, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
            # os.path.join(DATA_PATH, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
            # os.path.join(DATA_PATH, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
            # os.path.join(DATA_PATH, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
        ],
        "normalize": True,
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 64,
        "seed": config["seed"],
        "use_cache": False,
    },
    "skip_train": False,
    "skip_val": False,
    "skip_test": False,
    "normalize_neuron_coords": True,
    "average_test_multitrial": True,
    "save_test_multitrial": True,
    "test_batch_size": 7,
    "device": config["device"],
}

### comparison config
config["comparison"] = {
    "load_best": False,
    # "load_best": True,
    "eval_all_ckpts": True,
    # "eval_all_ckpts": False,
    # "find_best_ckpt_according_to": None,
    "find_best_ckpt_according_to": "SSIML-PL",
    "save_dir": None,
    "save_dir": os.path.join(
        "results",
        "transfer_learning",
    ),
    "loaded_ckpts_overwrite": True,
    "load_ckpts": None,
    # "load_ckpts": [
    #     {
    #         "path": os.path.join(
    #             "results",
    #             "transfer_learning",
    #             "2024-06-18_13-11-43.pt",
    #         ),
    #         "load_only": None, # load all
    #         # "load_only": ["Inverted Encoder"],
    #         "remap": None,
    #         # "remap": {"CNN-Conv (100% M-1 + 0% S-1)": "0% S-1"},
    #     },
    # ],
    "losses_to_plot": [
        "SSIML",
        "MSE",
        "PL",
        "FID",
    ],
    "syn_data_loss_curve": None,
    # "syn_data_loss_curve": {
    #     "losses_to_plot": ["SSIML", "MSE", "PL", "FID"],
    #     # "run_group_colors": None,
    #     "run_group_colors": ["#2066a8", "#81c3e4", "#568b87", "#80ae9a"],
    #     "mean_line_kwargs": {
    #         "linestyle": "-",
    #         "linewidth": 2.5,
    #         "label": "Mean",
    #         "color": "#dc353b",
    #     },
    #     "run_groups": {
    #         "CNN-Conv": {
    #             run_name: None for run_name in [
    #                 # "CNN-Conv (0%)",
    #                 # "CNN-Conv (25%)",
    #                 # "CNN-Conv (50%)",
    #                 # "CNN-Conv (87.5%)",
    #                 # "CNN-Conv (100%)",
    #                 r"CNN-Conv (0% S-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (25% S-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (50% S-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (87.5% S-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (100% S-All $\rightarrow$ M-1)",
    #             ]
    #         },
    #         "CNN-MEI": {
    #             run_name: None for run_name in [
    #                 # "CNN-MEI (0%)",
    #                 # "CNN-MEI (25%)",
    #                 # "CNN-MEI (50%)",
    #                 # "CNN-MEI (87.5%)",
    #                 # "CNN-MEI (100%)",
    #                 r"CNN-MEI (0% S-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (25% S-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (50% S-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (87.5% S-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (100% S-All $\rightarrow$ M-1)",
    #             ]
    #         },
    #         # "GAN-Conv": {
    #         #     run_name: None for run_name in [
    #         #         "GAN-Conv (0%)",
    #         #         "GAN-Conv (25%)",
    #         #         "GAN-Conv (50%)",
    #         #         "GAN-Conv (87.5%)",
    #         #         "GAN-Conv (100%)",
    #         #     ]
    #         # },
    #         # "GAN-MEI": {
    #         #     run_name: None for run_name in [
    #         #         "GAN-MEI (0%)",
    #         #         "GAN-MEI (25%)",
    #         #         "GAN-MEI (50%)",
    #         #         "GAN-MEI (87.5%)",
    #         #         "GAN-MEI (100%)",
    #         #     ]
    #         # },
    #     },
    # },
    # "syn_data_loss_curve": {
    #     "losses_to_plot": ["SSIML", "MSE", "PL", "FID"],
    #     # "run_group_colors": None,
    #     "run_group_colors": ["#2066a8", "#81c3e4", "#568b87", "#80ae9a"],
    #     "mean_line_kwargs": {
    #         "linestyle": "-",
    #         "linewidth": 2.5,
    #         "label": "Mean",
    #         "color": "#dc353b",
    #     },
    #     "run_groups": {
    #         "CNN-Conv": {
    #             run_name: None for run_name in [
    #                 r"CNN-Conv (0% C + 100% M-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (25% C + 75% M-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (50% C + 50% M-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (90% C + 10% M-All $\rightarrow$ M-1)",
    #                 r"CNN-Conv (100% C + 0% M-All $\rightarrow$ M-1)",
    #             ]
    #         },
    #         "CNN-MEI": {
    #             run_name: None for run_name in [
    #                 r"CNN-MEI (0% C + 100% M-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (25% C + 75% M-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (50% C + 50% M-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (90% C + 10% M-All $\rightarrow$ M-1)",
    #                 r"CNN-MEI (100% C + 0% M-All $\rightarrow$ M-1)",
    #             ]
    #         },
    #     },
    # },
    "plot_over_ckpts": None,
    # "plot_over_ckpts": {
    #     "to_plot": "SSIML",
    #     "max_epochs": 100,
    #     "conv_win": 3,
    # },
}


### Table 1
# config["comparison"]["to_compare"] = {
#     "Inverted Encoder": {
#         "decoder": InvertedEncoder(
#             encoder=get_encoder(
#                 ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
#                 device=config["device"],
#                 eval_mode=True,
#                 # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
#             ),
#             img_dims=(1, 36, 64),
#             stim_pred_init="zeros",
#             opter_cls=torch.optim.SGD,
#             opter_config={"lr": 50, "momentum": 0},
#             n_steps=500,
#             resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
#             stim_loss_fn=SSIMLoss(
#                 window=config["crop_win"],
#                 log_loss=True,
#                 inp_normalized=True,
#                 inp_standardized=False,
#             ),
#             img_gauss_blur_config=None,
#             img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 2},
#             device=config["device"],
#         ).to(config["device"]),
#         "run_name": None,
#     },
    
#     "CNN-FC (M-1)": {
#         "run_name": "2024-04-08_00-43-03",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_00-43-03", "decoder.pt"),
#     },
#     "CNN-FC w/ EM (M-1)": {
#         "run_name": "2024-04-25_19-39-26",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-25_19-39-26", "decoder.pt"),
#     },
#     "CNN-FC (M-All)": {
#         "run_name": "2024-04-08_00-39-27",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_00-39-27", "decoder.pt"),
#     },
#     "CNN-FC w/ EM (M-All)": {
#         "run_name": "2024-04-29_19-34-37",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-29_19-34-37", "decoder.pt"),
#     },
#     "CNN-Conv (M-1)": {
#         "run_name": "2024-03-27_11-35-11",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
#     },
#     "CNN-Conv w/ EM (M-1)": {
#         "run_name": "2024-04-11_10-22-00",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-11_10-22-00", "decoder.pt"),
#     },
#     "CNN-Conv (M-All)": {
#         "run_name": "2024-03-27_23-26-05",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-26-05", "decoder.pt"),
#     },
#     "CNN-Conv w/ EM (M-All)": {
#         "run_name": "2024-04-11_10-18-14",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-11_10-18-14", "decoder.pt"),
#     },
#     "CNN-MEI (M-1)": {
#         "run_name": "2024-04-09_08-42-29",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
#     },
#     "CNN-MEI w/ EM (M-1)": {
#         "run_name": "2024-04-12_23-44-06",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_23-44-06", "decoder.pt"),
#     },
#     "CNN-MEI (M-All)": {
#         "run_name": "2024-04-09_08-46-00",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-46-00", "decoder.pt"),
#     },
#     "CNN-MEI w/ EM (M-All)": {
#         "run_name": "2024-04-14_22-43-18",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-14_22-43-18", "decoder.pt"),
#     },
#     "GAN-Conv (M-1)": {
#         "run_name": "2024-04-10_11-06-28",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_11-06-28", "decoder.pt"),
#     },
#     "GAN-Conv w/ EM (M-1)": {
#         "run_name": "2024-04-11_13-54-42",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-11_13-54-42", "decoder.pt"),
#     },
#     "GAN-Conv (M-All)": {
#         "run_name": "2024-04-10_17-36-41",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_17-36-41", "decoder.pt"),
#     },
#     "GAN-Conv w/ EM (M-All)": {
#         "run_name": "2024-04-11_14-31-27",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-11_14-31-27", "decoder.pt"),
#     },
#     "GAN-MEI (M-1)": {
#         "run_name": "2024-04-12_11-19-16",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-19-16", "decoder.pt"),
#     },
#     "GAN-MEI w/ EM (M-1)": {
#         "run_name": "2024-04-23_13-46-11",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-23_13-46-11", "decoder.pt"),
#     },
#     "GAN-MEI (M-All)": {
#         "run_name": "2024-04-12_11-22-04",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-22-04", "decoder.pt"),
#     },
#     "GAN-MEI w/ EM (M-All)": {
#         "run_name": "2024-04-23_13-49-03",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-23_13-49-03", "decoder.pt"),
#     },
# }

### Table 2 (Trained on M-1/S-1)
# config["comparison"]["to_compare"] = {
#     "Inverted Encoder": {
#         "decoder": InvertedEncoder(
#             encoder=get_encoder(
#                 ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
#                 device=config["device"],
#                 eval_mode=True,
#                 # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
#             ),
#             img_dims=(1, 36, 64),
#             stim_pred_init="zeros",
#             opter_cls=torch.optim.SGD,
#             opter_config={"lr": 50, "momentum": 0},
#             n_steps=500,
#             resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
#             stim_loss_fn=SSIMLoss(
#                 window=config["crop_win"],
#                 log_loss=True,
#                 inp_normalized=True,
#                 inp_standardized=False,
#             ),
#             img_gauss_blur_config=None,
#             img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 2},
#             device=config["device"],
#         ).to(config["device"]),
#         "run_name": None,
#     },

#     "CNN-Conv (100% M-1 + 0% S-1)": {
#     # "0% S-1": {
#         "run_name": "2024-03-27_11-35-11",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
#     },
#     "CNN-Conv (75% M-1 + 25% S-1)": {
#     # "25% S-1": {
#         "run_name": "2024-03-27_23-16-33",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-16-33", "decoder.pt"),
#     },
#     "CNN-Conv (50% M-1 + 50% S-1)": {
#     # "50% S-1": {
#         "run_name": "2024-03-27_18-15-44",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_18-15-44", "decoder.pt"),
#     },
#     "CNN-Conv (12.5% M-1 + 87.5% S-1)": {
#     # "87.5% S-1": {
#         "run_name": "2024-04-08_21-11-50",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_21-11-50", "decoder.pt"),
#     },
#     "CNN-Conv (0% M-1 + 100% S-1)": {
#     # "100% S-1": {
#         "run_name": "2024-04-08_21-09-33",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_21-09-33", "decoder.pt"),
#     },
#     "CNN-MEI (100% M-1 + 0% S-1)": {
#         "run_name": "2024-04-09_08-42-29",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
#     },
#     "CNN-MEI (75% M-1 + 25% S-1)": {
#         "run_name": "2024-04-12_11-41-07",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-41-07", "decoder.pt"),
#     },
#     "CNN-MEI (50% M-1 + 50% S-1)": {
#         "run_name": "2024-04-12_11-26-43",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-26-43", "decoder.pt"),
#     },
#     "CNN-MEI (12.5% M-1 + 87.5% S-1)": {
#         "run_name": "2024-04-12_11-38-37",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-38-37", "decoder.pt"),
#     },
#     "CNN-MEI (0% M-1 + 100% S-1)": {
#         "run_name": "2024-04-12_11-31-42",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-31-42", "decoder.pt"),
#     },

#     "GAN-Conv (100% M-1 + 0% S-1)": {
#         "run_name": "2024-04-10_11-06-28",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_11-06-28", "decoder.pt"),
#     },
#     "GAN-Conv (75% M-1 + 25% S-1)": {
#         "run_name": "2024-04-17_20-06-17",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_20-06-17", "decoder.pt"),
#     },
#     "GAN-Conv (50% M-1 + 50% S-1)": {
#         "run_name": "2024-04-17_20-02-05",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_20-02-05", "decoder.pt"),
#     },
#     "GAN-Conv (12.5% M-1 + 87.5% S-1)": {
#         "run_name": "2024-04-18_09-10-28",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_09-10-28", "decoder.pt"),
#     },
#     "GAN-Conv (0% M-1 + 100% S-1)": {
#         "run_name": "2024-04-17_22-52-07",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_22-52-07", "decoder.pt"),
#     },
#     "GAN-MEI (100% M-1 + 0% S-1)": {
#         "run_name": "2024-04-12_11-19-16",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-19-16", "decoder.pt"),
#     },
#     "GAN-MEI (75% M-1 + 25% S-1)": {
#         "run_name": "2024-04-17_13-49-33",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_13-49-33", "decoder.pt"),
#     },
#     "GAN-MEI (50% M-1 + 50% S-1)": {
#         "run_name": "2024-04-17_13-41-46",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_13-41-46", "decoder.pt"),
#     },
#     "GAN-MEI (12.5% M-1 + 87.5% S-1)": {
#         "run_name": "2024-04-17_13-45-38",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_13-45-38", "decoder.pt"),
#     },
#     "GAN-MEI (0% M-1 + 100% S-1)": {
#         "run_name": "2024-04-17_19-57-26",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-17_19-57-26", "decoder.pt"),
#     },
# }

### Table 2 (Trained on M-All/S-All)
# config["comparison"]["to_compare"] = {
#     "Inverted Encoder": {
#         "decoder": InvertedEncoder(
#             encoder=get_encoder(
#                 ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
#                 device=config["device"],
#                 eval_mode=True,
#                 # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
#             ),
#             img_dims=(1, 36, 64),
#             stim_pred_init="zeros",
#             opter_cls=torch.optim.SGD,
#             opter_config={"lr": 50, "momentum": 0},
#             n_steps=500,
#             resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
#             stim_loss_fn=SSIMLoss(
#                 window=config["crop_win"],
#                 log_loss=True,
#                 inp_normalized=True,
#                 inp_standardized=False,
#             ),
#             img_gauss_blur_config=None,
#             img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 2},
#             device=config["device"],
#         ).to(config["device"]),
#         "run_name": None,
#     },
    
#     "CNN-Conv (100% M-All + 0% S-All)": {
#         "run_name": "2024-03-27_23-26-05",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-26-05", "decoder.pt"),
#     },
#     "CNN-Conv (75% M-All + 25% S-All)": {
#         "run_name": "2024-04-01_11-12-17",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-01_11-12-17", "decoder.pt"),
#     },
#     "CNN-Conv (50% M-All + 50% S-All)": {
#         "run_name": "2024-03-31_17-58-59",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-31_17-58-59", "decoder.pt"),
#     },
#     "CNN-Conv (12.5% M-All + 87.5% S-All)": {
#         "run_name": "2024-04-01_11-16-55",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-01_11-16-55", "decoder.pt"),
#     },
#     "CNN-Conv (0% M-All + 100% S-All)": {
#         "run_name": "2024-04-09_08-52-18",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-52-18", "decoder.pt"),
#     },
#     "CNN-MEI (100% M-All + 0% S-All)": {
#         "run_name": "2024-04-09_08-46-00",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-46-00", "decoder.pt"),
#     },
#     "CNN-MEI (75% M-All + 25% S-All)": {
#         "run_name": "2024-04-12_23-17-47",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_23-17-47", "decoder.pt"),
#     },
#     "CNN-MEI (50% M-All + 50% S-All)": {
#         "run_name": "2024-04-12_23-12-31",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_23-12-31", "decoder.pt"),
#     },
#     "CNN-MEI (12.5% M-All + 87.5% S-All)": {
#         "run_name": "2024-04-12_23-16-03",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_23-16-03", "decoder.pt"),
#     },
#     "CNN-MEI (0% M-All + 100% S-All)": {
#         "run_name": "2024-04-15_09-05-09",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-15_09-05-09", "decoder.pt"),
#     },

#     "GAN-Conv (100% M-All + 0% S-All)": {
#         "run_name": "2024-04-10_17-36-41",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_17-36-41", "decoder.pt"),
#     },
#     "GAN-Conv (75% M-All + 25% S-All)": {
#         "run_name": "2024-04-20_22-03-38",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-20_22-03-38", "decoder.pt"),
#     },
#     "GAN-Conv (50% M-All + 50% S-All)": {
#         "run_name": "2024-04-18_22-07-20",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_22-07-20", "decoder.pt"),
#     },
#     "GAN-Conv (12.5% M-All + 87.5% S-All)": {
#         "run_name": "2024-04-21_08-56-59",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-21_08-56-59", "decoder.pt"),
#     },
#     "GAN-Conv (0% M-All + 100% S-All)": {
#         "run_name": "2024-04-21_15-07-26",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-21_15-07-26", "decoder.pt"),
#     },
#     "GAN-MEI (100% M-All + 0% S-All)": {
#         "run_name": "2024-04-12_11-22-04",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-22-04", "decoder.pt"),
#     },
#     "GAN-MEI (75% M-All + 25% S-All)": {
#         "run_name": "2024-04-18_15-21-57",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_15-21-57", "decoder.pt"),
#     },
#     "GAN-MEI (50% M-All + 50% S-All)": {
#         "run_name": "2024-04-18_15-20-30",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_15-20-30", "decoder.pt"),
#     },
#     "GAN-MEI (12.5% M-All + 87.5% S-All)": {
#         "run_name": "2024-04-18_15-23-39",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_15-23-39", "decoder.pt"),
#     },
#     "GAN-MEI (0% M-All + 100% S-All)": {
#         "run_name": "2024-04-19_08-08-05",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-19_08-08-05", "decoder.pt"),
#     },
# }

### Table 4 - Transfer learning
config["comparison"]["to_compare"] = {
    "Inverted Encoder": {
        "decoder": InvertedEncoder(
            encoder=get_encoder(
                ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
                device=config["device"],
                eval_mode=True,
                # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
            ),
            img_dims=(1, 36, 64),
            stim_pred_init="zeros",
            opter_cls=torch.optim.SGD,
            opter_config={"lr": 50, "momentum": 0},
            n_steps=500,
            resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
            stim_loss_fn=SSIMLoss(
                window=config["crop_win"],
                log_loss=True,
                inp_normalized=True,
                inp_standardized=False,
            ),
            img_gauss_blur_config=None,
            img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 2.},
            device=config["device"],
        ).to(config["device"]),
        "run_name": None,
    },

    ### No pretraining
    "CNN-Conv (M-1)": {
        "run_name": "2024-03-27_11-35-11",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
    },
    "CNN-MEI (M-1)": {
        "run_name": "2024-04-09_08-42-29",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
    },
    "GAN-Conv (M-1)": {
        "run_name": "2024-04-10_11-06-28",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_11-06-28", "decoder.pt"),
    },
    "GAN-MEI (M-1)": {
        "run_name": "2024-04-12_11-19-16",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-19-16", "decoder.pt"),
    },

    ### C + M-All -> M-1
    r"CNN-Conv (0% C + 100% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-10_17-54-33",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-10_17-54-33", "decoder.pt"),
        "syn_data_percentage": 0,
    },
    r"CNN-Conv (25% C + 75% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-16_09-26-19",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-16_09-26-19", "decoder.pt"),
        "syn_data_percentage": 25,
    },
    r"CNN-Conv (50% C + 50% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-15_08-58-54",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-15_08-58-54", "decoder.pt"),
        "syn_data_percentage": 50,
    },
    r"CNN-Conv (50% C + 50% M-All $\rightarrow$ M-1) w/ SSIML-PL": {
        "run_name": "2024-06-15_09-12-25",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-15_09-12-25", "decoder.pt"),
        "syn_data_percentage": 50,
    },
    r"CNN-Conv (90% C + 10% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-15_09-03-51",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-15_09-03-51", "decoder.pt"),
        "syn_data_percentage": 90,
    },
    r"CNN-Conv (90% C + 10% M-All $\rightarrow$ M-1) w/ SSIML-PL": {
        "run_name": "2024-06-15_09-05-33",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-15_09-05-33", "decoder.pt"),
        "syn_data_percentage": 90,
    },
    r"CNN-Conv (100% C + 0% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-17_09-00-13",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-17_09-00-13", "decoder.pt"),
        "syn_data_percentage": 100,
    },
    r"CNN-MEI (0% C + 100% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-17_17-31-11",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-17_17-31-11", "decoder.pt"),
        "syn_data_percentage": 0,
    },
    r"CNN-MEI (25% C + 75% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-17_08-15-24",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-17_08-15-24", "decoder.pt"),
        "syn_data_percentage": 25,
    },
    r"CNN-MEI (50% C + 50% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-17_08-18-26",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-17_08-18-26", "decoder.pt"),
        "syn_data_percentage": 50,
    },
    r"CNN-MEI (90% C + 10% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-17_08-20-32",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-17_08-20-32", "decoder.pt"),
        "syn_data_percentage": 90,
    },
    r"CNN-MEI (100% C + 0% M-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-18_07-42-09",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-18_07-42-09", "decoder.pt"),
        "syn_data_percentage": 100,
    },

    ### C + M-1 -> M-1
    r"CNN-Conv (50% C + 50% M-1 $\rightarrow$ M-1)": {
        "run_name": "2024-05-19_13-27-50",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-19_13-27-50", "decoder.pt"),
    },
    r"CNN-Conv (90% C + 10% M-1 $\rightarrow$ M-1)": {
        "run_name": "2024-05-19_11-02-33",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-19_11-02-33", "decoder.pt"),
    },

    ### C -> M-All

    ### S-All -> M-1
    r"CNN-Conv (0% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-10_17-54-33",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-10_17-54-33", "decoder.pt"),
        "syn_data_percentage": 0,
    },
    r"CNN-Conv (25% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-18_16-19-57",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-18_16-19-57", "decoder.pt"),
        "syn_data_percentage": 25,
    },
    r"CNN-Conv (50% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-10_22-34-23",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-10_22-34-23", "decoder.pt"),
        "syn_data_percentage": 50,
    },
    r"CNN-Conv (87.5% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-10_18-00-10",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-10_18-00-10", "decoder.pt"),
        "syn_data_percentage": 87.5,
    },
    r"CNN-Conv (100% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-10_22-32-05",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-10_22-32-05", "decoder.pt"),
        "syn_data_percentage": 100,
    },
    r"CNN-MEI (0% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-17_17-31-11",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-17_17-31-11", "decoder.pt"),
        "syn_data_percentage": 0,
    },
    r"CNN-MEI (25% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-06-18_16-17-53",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-06-18_16-17-53", "decoder.pt"),
        "syn_data_percentage": 25,
    },
    r"CNN-MEI (50% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-28_19-14-21",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-28_19-14-21", "decoder.pt"),
        "syn_data_percentage": 50,
    },
    r"CNN-MEI (87.5% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-28_19-16-46",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-28_19-16-46", "decoder.pt"),
        "syn_data_percentage": 87.5,
    },
    r"CNN-MEI (100% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-28_19-18-33",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-28_19-18-33", "decoder.pt"),
        "syn_data_percentage": 100,
    },

    r"GAN-Conv (0% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-04-11_10-41-27",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-11_10-41-27", "decoder.pt"),
    },
    r"GAN-Conv (50% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_07-38-17",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_07-38-17", "decoder.pt"),
    },
    r"GAN-Conv (87.5% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_07-40-05",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_07-40-05", "decoder.pt"),
    },
    r"GAN-Conv (100% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_07-42-24",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_07-42-24", "decoder.pt"),
    },
    r"GAN-MEI (0% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_09-30-53",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_09-30-53", "decoder.pt"),
    },
    r"GAN-MEI (50% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_09-32-21",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_09-32-21", "decoder.pt"),
    },
    r"GAN-MEI (87.5% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_09-33-48",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_09-33-48", "decoder.pt"),
    },
    r"GAN-MEI (100% S-All $\rightarrow$ M-1)": {
        "run_name": "2024-05-13_09-35-03",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-13_09-35-03", "decoder.pt"),
    },

    r"GAN-MEI (100% S-All $\rightarrow$ M-1, SSIML-PL fine-tuning)*": {
        "run_name": "2024-05-19_22-13-01",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-05-19_22-13-01", "ckpt/decoder_141.pt"),
        "skip_model_selection": True,
    },
    r"CNN-MEI (C $\rightarrow$ M-1)*": {
        "run_name": "2024-04-26_21-54-43",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-26_21-54-43", "ckpt/decoder_75.pt"),
        "skip_model_selection": True,
    },
    r"CNN-MEI (C $\rightarrow$ M-All)*": {
        "run_name": "2024-04-26_21-51-47",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-26_21-51-47", "ckpt/decoder_50.pt"),
        "skip_model_selection": True,
    },
    r"GAN-MEI w/ EM (M-1)*": {
        "run_name": "2024-04-23_13-46-11",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-23_13-46-11", "ckpt/decoder_175.pt"),
        "skip_model_selection": True,
    },
    r"GAN-MEI (M-1)*": {
        "run_name": "2024-04-12_11-19-16",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-19-16", "ckpt/decoder_40.pt"),
        "skip_model_selection": True,
    },
}

### Contrastive regularization
# config["comparison"]["to_compare"] = {
#     "Inverted Encoder": {
#         "decoder": InvertedEncoder(
#             encoder=get_encoder(
#                 ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
#                 device=config["device"],
#                 eval_mode=True,
#                 # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
#             ),
#             img_dims=(1, 36, 64),
#             stim_pred_init="zeros",
#             opter_cls=torch.optim.SGD,
#             opter_config={"lr": 50, "momentum": 0},
#             n_steps=500,
#             resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
#             stim_loss_fn=SSIMLoss(
#                 window=config["crop_win"],
#                 log_loss=True,
#                 inp_normalized=True,
#                 inp_standardized=False,
#             ),
#             img_gauss_blur_config=None,
#             img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 2.},
#             device=config["device"],
#         ).to(config["device"]),
#         "run_name": None,
#     },

#     r"CNN-MEI (M-1)": {
#         "run_name": "2024-04-09_08-42-29",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
#     },
#     r"CNN-MEI w/ CR (M-1)": {
#         "run_name": "2024-05-26_23-06-36",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-26_23-06-36", "decoder.pt"),
#     },
#     r"CNN-MEI (M-1 + noisy M-1)": {
#         "run_name": "2024-05-26_23-26-58",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-26_23-26-58", "decoder.pt"),
#     },
#     r"CNN-MEI w/ CR (M-1 + noisy M-1)": {
#         "run_name": "2024-05-26_23-25-12",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-26_23-25-12", "decoder.pt"),
#     },
#     r"CNN-MEI w/ CRD (M-1 + noisy M-1)": {
#         "run_name": "2024-05-27_23-10-37",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-27_23-10-37", "decoder.pt"),
#     },
#     r"CNN-MEI w/ CR (M-1 + S-1)": {
#         "run_name": "2024-05-26_23-33-12",
#         "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-05-26_23-33-12", "decoder.pt"),
#     },
# }


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")

    if config["comparison"]["load_best"] and config["comparison"]["eval_all_ckpts"]:
        print("[WARNING] both the eval_all_ckpts and load_best are set to True - still loading current (not the best) decoders.")
    assert config["comparison"]["eval_all_ckpts"] is True or config["comparison"]["find_best_ckpt_according_to"] is None
    assert config["comparison"]["find_best_ckpt_according_to"] is None or config["comparison"]["load_best"] is False
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### get data samples
    dataloaders, neuron_coords = get_mouse_v1_data(config["data"])
    sample_data_key = dataloaders["mouse_v1"]["test"].data_keys[0]
    datapoint = next(iter(dataloaders["mouse_v1"]["test"].dataloaders[0]))
    stim, resp, pupil_center = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"]), datapoint.pupil_center.to(config["device"])
    
    ### prepare comparison
    runs_to_compare = dict()
    if config["comparison"]["load_ckpts"] is not None:
        for ckpt_config in config["comparison"]["load_ckpts"]:
            print(f"Loading checkpoint from {ckpt_config['path']}...")
            loaded_runs = torch.load(ckpt_config["path"], map_location=config["device"], pickle_module=dill)["runs"]
            if ckpt_config["load_only"] is not None:
                ### load only selected runs
                runs_to_compare.update({run_name: loaded_runs[run_name] for run_name in ckpt_config["load_only"]})
            else:
                ### load all
                runs_to_compare.update(loaded_runs)
            print(f"[INFO] Loaded from ckpt: {', '.join(list(runs_to_compare.keys()))}")
            
            ### remap names
            remap = ckpt_config["remap"]
            if remap is not None:
                for in_name, out_name in remap.items():
                    if in_name not in runs_to_compare:
                        continue
                    runs_to_compare[out_name] = runs_to_compare[in_name]
                    del runs_to_compare[in_name]
                print(f"[INFO] Remapped from ckpt to: {', '.join(list(runs_to_compare.keys()))}")

    ### merge and reorder with current to_compare config
    _runs_to_compare = dict()
    for run_name in config["comparison"]["to_compare"].keys():
        if run_name in runs_to_compare \
            and config["comparison"]["loaded_ckpts_overwrite"] \
            and config["comparison"]["to_compare"][run_name]["run_name"] == runs_to_compare[run_name]["run_name"]:
            _runs_to_compare[run_name] = runs_to_compare[run_name]
            _runs_to_compare[run_name].update(config["comparison"]["to_compare"][run_name]) # keep the key-value pairs in the current config
        else:
            _runs_to_compare[run_name] = config["comparison"]["to_compare"][run_name]
    runs_to_compare = _runs_to_compare
    metrics = get_metrics(crop_win=config["crop_win"], device=config["device"])

    ### load and compare models
    for k in runs_to_compare.keys():
        print(f"Loading {k} model from ckpt (run name: {runs_to_compare[k]['run_name']})...")
        if "test_losses" in runs_to_compare[k]:
            print(f"  Skipping...")
            continue
        run_dict = runs_to_compare[k]
        run_name = run_dict["run_name"]
        for _k in ("test_losses", "configs", "histories", "best_val_losses", "stim_pred_best", "ckpt_paths"):
            run_dict[_k] = []

        ### get filepath to the model ckpt
        if "decoder" in run_dict and run_dict["decoder"] is not None:
            ### decoder already prepared
            run_dict["ckpt_paths"].append(None)
        else:
            if config["comparison"]["eval_all_ckpts"] \
                and ("skip_model_selection" not in run_dict or not run_dict["skip_model_selection"]):
                ckpts_dir = os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt")
                run_dict["ckpt_paths"].extend([os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt", ckpt_name) for ckpt_name in os.listdir(ckpts_dir)])

                ### find best ckpt according to some metric
                print(f"  Finding the best ckpt according to {config['comparison']['find_best_ckpt_according_to']}...")
                best_ckpt_path, _, all_ckpts_losses = find_best_ckpt(config=config, ckpt_paths=run_dict["ckpt_paths"], metrics=metrics)
                run_dict["all_ckpts_losses"] = all_ckpts_losses
                run_dict["ckpt_paths"] = [best_ckpt_path]
            else:
                run_dict["ckpt_paths"].append(run_dict["ckpt_path"])

        ### eval ckpts
        print(f"  Evaluating ckpts on the test set...")
        for ckpt_path in run_dict["ckpt_paths"]:
            if "decoder" in run_dict and run_dict["decoder"] is not None:
                print(f"  Using {k} model from run_dict...")
                decoder = run_dict["decoder"]
                ckpt = None
            else:
                ### load ckpt and init
                decoder, ckpt = load_decoder_from_ckpt(
                    ckpt_path=ckpt_path,
                    load_best=config["comparison"]["load_best"] and not config["comparison"]["eval_all_ckpts"],
                    device=config["device"],
                )
                run_dict["configs"].append(ckpt["config"])
                run_dict["histories"].append(ckpt["history"])
                run_dict["best_val_losses"].append(ckpt["best"]["val_loss"])

            ### get reconstructions
            if decoder.__class__.__name__ == "InvertedEncoder":
                stim_pred_best, _, _ = decoder(resp, stim, additional_encoder_inp={
                    "data_key": sample_data_key,
                    "pupil_center": pupil_center,
                })
                stim_pred_best = stim_pred_best.detach().cpu()
            else:
                stim_pred_best = decoder(resp, data_key=sample_data_key, neuron_coords=neuron_coords[sample_data_key], pupil_center=pupil_center).detach().cpu()

            ### eval
            dls, neuron_coords = get_all_data(config=config)
            run_dict["test_losses"].append(eval_decoder(
                model=decoder,
                dataloader=dls["mouse_v1"]["test"],
                loss_fns=metrics,
                config=config,
                calc_fid="FID" in config["comparison"]["losses_to_plot"],
            ))

            run_dict["stim_pred_best"].append(stim_pred_best.detach().cpu())

    ### save the results
    if config["comparison"]["save_dir"]:
        print(f"Saving the results to {config['comparison']['save_dir']}")
        os.makedirs(config["comparison"]["save_dir"], exist_ok=True)
        torch.save({
                "runs": runs_to_compare,
                "config": config,
            },
            os.path.join(config["comparison"]["save_dir"], f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"),
            pickle_module=dill,
        )

    ### plotting
    print(f"Plotting...")

    # plot reconstructions
    for f_type in ("png", "pdf"):
        # all together
        plot_reconstructions(
            runs=runs_to_compare,
            stim=stim,
            stim_label="Target",
            manually_standardize=True,
            crop_win=config["crop_win"],
            save_to=os.path.join(config["comparison"]["save_dir"], f"reconstructions.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )
        # individually
        os.makedirs(os.path.join(config["comparison"]["save_dir"], "individual_reconstructions"), exist_ok=True)
        os.makedirs(os.path.join(config["comparison"]["save_dir"], "individual_reconstructions", "png"), exist_ok=True)
        os.makedirs(os.path.join(config["comparison"]["save_dir"], "individual_reconstructions", "pdf"), exist_ok=True)
        for k in runs_to_compare:
            plot_reconstructions(
                runs={k: runs_to_compare[k]},
                stim=stim,
                stim_label="Target",
                manually_standardize=True,
                crop_win=config["crop_win"],
                save_to=os.path.join(config["comparison"]["save_dir"], "individual_reconstructions", f_type, f"{slugify(k)}.{f_type}") \
                    if config["comparison"]["save_dir"] else None,
            )

    # plot metrics
    for f_type in ("png", "pdf"):
        plot_metrics(
            runs_to_compare=runs_to_compare,
            losses_to_plot=config["comparison"]["losses_to_plot"],
            bar_width=0.7,
            save_to=os.path.join(config["comparison"]["save_dir"], f"metrics.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )

    # plot test losses based on syn. data percentage
    if config["comparison"]["syn_data_loss_curve"] is not None:
        run_groups = config["comparison"]["syn_data_loss_curve"]["run_groups"]
        ### fill-in the results
        for run_group_name in run_groups.keys():
            for run_name in run_groups[run_group_name]:
                run_groups[run_group_name][run_name] = runs_to_compare[run_name]
        for f_type in ("png", "pdf"):
            plot_syn_data_loss_curve(
                run_groups=run_groups,
                losses_to_plot=config["comparison"]["syn_data_loss_curve"]["losses_to_plot"],
                run_group_colors=config["comparison"]["syn_data_loss_curve"]["run_group_colors"],
                mean_line_kwargs=config["comparison"]["syn_data_loss_curve"]["mean_line_kwargs"],
                xlabel="% of synthetic data",
                save_to=os.path.join(config["comparison"]["save_dir"], f"syn_data_loss_curve.{f_type}") \
                    if config["comparison"]["save_dir"] else None,
            )

    # plot perfomrance over ckpts
    if "plot_over_ckpts" in config["comparison"] and config["comparison"]["plot_over_ckpts"] is not None:
        for f_type in ("png", "pdf"):
            plot_over_ckpts(
                runs=runs_to_compare,
                to_plot=config["comparison"]["plot_over_ckpts"]["to_plot"],
                max_epochs=config["comparison"]["plot_over_ckpts"]["max_epochs"],
                conv_win=config["comparison"]["plot_over_ckpts"]["conv_win"],
                save_to=os.path.join(config["comparison"]["save_dir"], f"{config['comparison']['plot_over_ckpts']['to_plot']}_over_ckpts.{f_type}") \
                    if config["comparison"]["save_dir"] else None,
            )

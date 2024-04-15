import os
os.environ["DATA_PATH"] = "/home/sobotj11/decoding-brain-activity/data"
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
from csng.readins import (
    MultiReadIn,
    HypernetReadIn,
    ConvReadIn,
    AttentionReadIn,
    FCReadIn,
    AutoEncoderReadIn,
    Conv1dReadIn,
)

# from BoostedInvertedEncoder import BoostedInvertedEncoder
from encoder import get_encoder
from data_utils import get_mouse_v1_data, PerSampleStoredDataset, append_syn_dataloaders, append_data_aug_dataloaders
from comparison_utils import (
    load_decoder_from_ckpt,
    get_metrics,
    find_best_ckpt,
    eval_decoder,
    get_all_data,
    plot_reconstructions,
    plot_metrics,
    plot_over_training,
    plot_reconstructions_publication,
    plot_metrics_publication,
)

lt.monkey_patch()

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
print(f"{DATA_PATH=}")


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
        "batch_size": 128,
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
    "find_best_ckpt_according_to": None,
    "find_best_ckpt_according_to": "Perceptual Loss (VGG16)",
    "find_best_ckpt_according_to": "SSIML + PSL",
    "save_dir": None,
    "save_dir": os.path.join(
        "results",
        "table_02",
    ),
    "load_ckpt": None,
    # "load_ckpt": {
    #     "path": os.path.join(
    #         "results",
    #         "fig1",
    #         "2024-04-08_15-23-37.pt",
    #     ),
    # },
    # "load_ckpt": {
    #     "path": "encoder_inversion_eval_all_mice_26-03-24.pt",
    #     "path": "encoder_inversion_eval_mouse_1_12-04-24.pt",
    # },
}

config["comparison"]["to_compare"] = {
    # {
    #     "decoder": InvertedEncoder(
    #         encoder=encoder,
    #         img_dims=(1,36,64),
    #         stim_pred_init="zeros",
    #         opter_cls=torch.optim.SGD,
    #         opter_config={"lr": 1500, "momentum": 0},
    #         n_steps=400,
    #         resp_loss_fn=F.mse_loss,
    #         stim_loss_fn=SSIMLoss(
    #             window=config["crop_win"],
    #             log_loss=True,
    #             inp_normalized=True,
    #             inp_standardized=False,
    #         ),
    #         img_gauss_blur_config=None,
    #         img_grad_gauss_blur_config={"kernel_size": 17, "sigma": 2},
    #         device=config["device"],
    #     ).to(config["device"]),
    #     "run_name": None,
    # },


    ### Table 1 (Initial comparison)
    # "CNN-FC (M-1)": {
    #     "run_name": "2024-04-08_00-43-03",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_00-43-03", "decoder.pt"),
    # },
    # "CNN-Conv (M-1)": {
    #     "run_name": "2024-03-27_11-35-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
    # },
    # "CNN-MEI (M-1)": {
    #     "run_name": "2024-04-09_08-42-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
    # },
    # "CNN-FC (M-All)": {
    #     "run_name": "2024-04-08_00-39-27",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_00-39-27", "decoder.pt"),
    # },
    # "CNN-Conv (M-All)": {
    #     "run_name": "2024-03-27_23-26-05",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-26-05", "decoder.pt"),
    # },
    # "CNN-MEI (M-All)": {
    #     "run_name": "2024-04-09_08-46-00",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-46-00", "decoder.pt"),
    # },
    # "GAN (M-1)": {
    #     "run_name": "2024-04-10_11-06-28",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_11-06-28", "decoder.pt"),
    # },
    # "GAN (M-All)": {
    #     "run_name": "2024-04-10_17-36-41",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_17-36-41", "decoder.pt"),
    # },


    ### Table 2 (Encoder matching)
    "CNN-Conv (M-1)": {
        "run_name": "2024-03-27_11-35-11",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
    },
    "CNN-Conv w/ encoder matching (M-1)": {
        "run_name": "2024-04-11_10-22-00",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-11_10-22-00", "decoder.pt"),
    },
    "CNN-Conv (M-All)": {
        "run_name": "2024-03-27_23-26-05",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-26-05", "decoder.pt"),
    },
    "CNN-Conv w/ encoder matching (M-All)": {
        "run_name": "2024-04-11_10-18-14",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-11_10-18-14", "decoder.pt"),
    },
    "CNN-MEI (M-1)": {
        "run_name": "2024-04-09_08-42-29",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
    },
    "CNN-MEI w/ encoder matching (M-1)": {
        "run_name": "2024-04-12_23-44-06",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_23-44-06", "decoder.pt"),
    },
    "GAN (M-1)": {
        "run_name": "2024-04-10_11-06-28",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_11-06-28", "decoder.pt"),
    },
    "GAN w/ encoder matching (M-1)": {
        "run_name": "2024-04-11_13-54-42",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-11_13-54-42", "decoder.pt"),
    },
    "GAN (M-All)": {
        "run_name": "2024-04-10_17-36-41",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-10_17-36-41", "decoder.pt"),
    },
    "GAN w/ encoder matching (M-All)": {
        "run_name": "2024-04-11_14-31-27",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-11_14-31-27", "decoder.pt"),
    },

    ### Table 3 (Synthetic data M-1/S-1)
    # "CNN-Conv (0%)": {
    #     "run_name": "2024-03-27_11-35-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_11-35-11", "decoder.pt"),
    # },
    # "CNN-Conv (25%)": {
    #     "run_name": "2024-03-27_23-16-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_23-16-33", "decoder.pt"),
    # },
    # "CNN-Conv (50%)": {
    #     "run_name": "2024-03-27_18-15-44",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-03-27_18-15-44", "decoder.pt"),
    # },
    # "CNN-Conv (87.5%)": {
    #     "run_name": "2024-04-08_21-11-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_21-11-50", "decoder.pt"),
    # },
    # "CNN-Conv (100%)": {
    #     "run_name": "2024-04-08_21-09-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-08_21-09-33", "decoder.pt"),
    # },
    # "CNN-MEI (0%)": {
    #     "run_name": "2024-04-09_08-42-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-09_08-42-29", "decoder.pt"),
    # },
    # "CNN-MEI (25%)": {
    #     "run_name": "2024-04-12_11-41-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-41-07", "decoder.pt"),
    # },
    # "CNN-MEI (50%)": {
    #     "run_name": "2024-04-12_11-26-43",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-26-43", "decoder.pt"),
    # },
    # "CNN-MEI (87.5%)": {
    #     "run_name": "2024-04-12_11-38-37",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-38-37", "decoder.pt"),
    # },
    # "CNN-MEI (100%)": {
    #     "run_name": "2024-04-12_11-31-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-12_11-31-42", "decoder.pt"),
    # },
}


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
    if config["comparison"]["load_ckpt"] is not None:
        print(f"Loading checkpoint from {config['comparison']['load_ckpt']['path']}...")
        runs_to_compare.update(
            torch.load(config["comparison"]["load_ckpt"]["path"], map_location=config["device"])
        )
    runs_to_compare.update(config["comparison"]["to_compare"])
    metrics = get_metrics(config=config)

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

        if "decoder" in run_dict and run_dict["decoder"] is not None:
            run_dict["ckpt_paths"].append(None)
        else:
            if config["comparison"]["eval_all_ckpts"]:
                ckpts_dir = os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt")
                run_dict["ckpt_paths"].extend([os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt", ckpt_name) for ckpt_name in os.listdir(ckpts_dir)])
            else:
                run_dict["ckpt_paths"].append(run_dict["ckpt_path"])

            ### find best ckpt according to some metric
            if config["comparison"]["find_best_ckpt_according_to"] is not None:
                print(f"  Finding the best ckpt according to {config['comparison']['find_best_ckpt_according_to']}...")
                run_dict["ckpt_paths"] = [find_best_ckpt(config=config, ckpt_paths=run_dict["ckpt_paths"], metrics=metrics)[0]]

        ### eval ckpts
        print(f"  Evaluating ckpts on the test set...")
        for ckpt_path in run_dict["ckpt_paths"]:
            if "decoder" in run_dict and run_dict["decoder"] is not None:
                print(f"  Using {k} model from run_dict...")
                decoder = run_dict["decoder"]
                ckpt = None
            else:
                ### load ckpt and init
                decoder, ckpt = load_decoder_from_ckpt(config=config, ckpt_path=ckpt_path)
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
                normalize_decoded=False,
                config=config,
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
            os.path.join(config["comparison"]["save_dir"], f"M-1_PSL_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"),
            pickle_module=dill,
        )

    ### plotting
    print(f"Plotting...")
    ### plot reconstructions
    for f_type in ("png", "pdf"):
        plot_reconstructions_publication(
            runs_to_compare=runs_to_compare,
            stim=stim,
            config=config,
            save_to=os.path.join(config["comparison"]["save_dir"], f"reconstructions.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )

    ### plot metrics
    for f_type in ("png", "pdf"):
        plot_metrics_publication(
            runs_to_compare=runs_to_compare,
            losses_to_plot=[
                "SSIM",
                # "SSIM Loss",
                # "Log SSIM Loss",
                # "MultiSSIM Loss",
                # "Log MultiSSIM Loss",
                "MSE",
                # "MAE",
                "FFL",
                "Perceptual Loss (VGG16)",
                # "Perceptual Loss (Encoder)",
            ],
            bar_width=0.7,
            save_to=os.path.join(config["comparison"]["save_dir"], f"metrics.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )

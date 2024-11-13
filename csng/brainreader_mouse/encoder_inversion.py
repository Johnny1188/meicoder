import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import lovely_tensors as lt
import dill
import itertools

import csng
from csng.Ensemble import EnsembleInvEnc
from csng.InvertedEncoder import InvertedEncoder, InvertedEncoderBrainreader
from csng.utils import crop, plot_comparison, dict_to_str, standardize, normalize, count_parameters, slugify, seed_all
from csng.comparison import eval_decoder, get_metrics
from csng.brainreader_mouse.encoder import get_encoder
from csng.brainreader_mouse.data import get_brainreader_data

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "brainreader")


### prepare config
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "device": os.environ["DEVICE"],
    "seed": 0,
    "crop_win": (36, 64),
}

### data config
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    # "max_batches": 15,
    "max_batches": 10,
    "data_dir": os.path.join(DATA_PATH, "data"),
    "batch_size": 32,
    # "sessions": list(range(1, 2)),
    "sessions": [6],
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}


### encoder inversion config
config["enc_inv"] = {
    "model": {
        ### InvertedEncoderBrainreader
        # "encoder": get_encoder(
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_mall.pth"),
        #     eval_mode=True,
        #     device=config["device"],
        # ),
        # "img_dims": (1, 36, 64),
        # "stim_pred_init": "randn",
        # "lr": 100,
        # "n_steps": 1000,
        # "img_grad_gauss_blur_sigma": 2,
        # "jitter": 0,
        # "mse_reduction": "per_sample_mean_sum",
        # "device": config["device"],
        
        ### InvertedEncoder
        # "encoder": get_encoder(
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_mall.pth"),
        #     eval_mode=True,
        #     device=config["device"],
        # ),
        # "img_dims": (1, 36, 64),
        # "stim_pred_init": "zeros",
        # "opter_config": {"lr": 100},
        # "n_steps": 1000,
        # "img_gauss_blur_config": None,
        # "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2},
        # "device": config["device"],
        
        ### EnsembleInvEnc
        "encoder_paths": [
            os.path.join(DATA_PATH, "models", "encoder_m6_seed0.pth"),
            os.path.join(DATA_PATH, "models", "encoder_m6_seed1.pth"),
            os.path.join(DATA_PATH, "models", "encoder_m6_seed2.pth"),
            os.path.join(DATA_PATH, "models", "encoder_m6_seed3.pth"),
            os.path.join(DATA_PATH, "models", "encoder_m6_seed4.pth"),
        ],
        "encoder_config": {
            "img_dims": (1, 36, 64),
            "stim_pred_init": "randn",
            "lr": 3000,
            "n_steps": 1000,
            "img_grad_gauss_blur_sigma": 2.,
            "jitter": 0,
            "mse_reduction": "per_sample_mean_sum",
            "device": config["device"],
        },
        "use_brainreader_encoder": True,
        "device": config["device"],
    },
    "sample_from": "test",
    "eval_on": "val",
    # "eval_on": None,
    "loss_fns": get_metrics(crop_win=config["crop_win"], device=config["device"]),
    "save_dir": os.path.join(DATA_PATH, "models", "inverted_encoder"),
    # "find_best_ckpt_according_to": "SSIML-PL",
    "find_best_ckpt_according_to": "FID",
    "max_batches": None,
}

### hyperparam runs config - either manually selected or grid search
config_updates = [dict()]
### EnsembleInvEnc
# config_updates = [
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 500,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 1.5,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 500,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 2,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 500,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 2.5,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1500,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 2,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1000,
#         "n_steps": 500,
#         "img_grad_gauss_blur_sigma": 2,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1000,
#         "n_steps": 1500,
#         "img_grad_gauss_blur_sigma": 2,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1000,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 1.5,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1000,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 2,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
#     {"encoder_config": {
#         "img_dims": (1, 36, 64),
#         "stim_pred_init": "randn",
#         "lr": 1000,
#         "n_steps": 1000,
#         "img_grad_gauss_blur_sigma": 2.5,
#         "jitter": 0,
#         "mse_reduction": "per_sample_mean_sum",
#         "device": config["device"],
#     }},
# ]
config_grid_search = None
# config_grid_search = {
#     ### InvertedEncoderBrainreader
#     # "n_steps": [200, 600, 1000],
#     "n_steps": [1000],
#     # "lr": [100, 250, 500, 1000],
#     "lr": [500, 1000],
#     # "img_grad_gauss_blur_sigma": [1.5, 2, 2.5],
#     "img_grad_gauss_blur_sigma": [2, 2.5],
#     "jitter": [0],
    
#     ### InvertedEncoder
#     # "n_steps": [300, 600, 1000],
#     # "opter_config": [{"lr": 500}, {"lr": 1000}],
#     # "img_grad_gauss_blur_config": [{"kernel_size": 13, "sigma": 1.5}, {"kernel_size": 13, "sigma": 2}, {"kernel_size": 13, "sigma": 2.5}],
# }


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    seed_all(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["enc_inv"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["enc_inv"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dls = get_brainreader_data(config=config["data"]["brainreader_mouse"])
    sample_data_key = dls["brainreader_mouse"][config["enc_inv"]["sample_from"]].data_keys[0]
    datapoint = next(iter(dls["brainreader_mouse"][config["enc_inv"]["sample_from"]].dataloaders[0]))
    stim, resp = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"])

    ### prepare config_updates
    if config_grid_search is not None:
        keys, vals = zip(*config_grid_search.items())
        config_updates.extend([dict(zip(keys, v)) for v in itertools.product(*vals)])
    print(f"[INFO] Config updates to try:\n ", "\n  ".join([dict_to_str(config_update) for config_update in config_updates]))

    ### run
    best = {"config": None, "val_loss": np.inf, "idx": None}
    print(f"[INFO] Hyperparameter search starts.")
    for i, config_update in enumerate(config_updates):
        print(f" [{i}/{len(config_updates)}]", end="")

        ### setup the model
        run_config = deepcopy(config)
        run_config["enc_inv"]["model"].update(config_update)
        run_name = f"{i}__{slugify(config_update)}"
        model = EnsembleInvEnc(**run_config["enc_inv"]["model"]).to(config["device"])
        # model = InvertedEncoderBrainreader(**run_config["enc_inv"]["model"]).to(config["device"])
        # model = InvertedEncoder(**run_config["enc_inv"]["model"]).to(config["device"])

        ### eval
        if config["enc_inv"]["eval_on"] is not None:
            dls = get_brainreader_data(config["data"]["brainreader_mouse"])
            val_losses = eval_decoder(
                model=model,
                dataloaders={"brainreader_mouse": dls["brainreader_mouse"][config["enc_inv"]["eval_on"]]},
                loss_fns=config["enc_inv"]["loss_fns"],
                config=config,
                calc_fid=config["enc_inv"]["find_best_ckpt_according_to"] == "FID",
                max_batches=config["enc_inv"]["max_batches"],
            )

            ### update best
            val_loss = val_losses[config["enc_inv"]["find_best_ckpt_according_to"]]["total"]
            print(f"  val_loss={val_loss:.3f}", end="")
            if val_loss < best["val_loss"]:
                print(f" >>> new best", end="")
                best["val_loss"] = val_loss
                best["config"] = run_config
                best["idx"] = i
            print("")
            print(f"   {slugify(config_update)}")

        ### plot sample
        stim_pred = model(
            resp=resp,
            data_key=sample_data_key,
        )
        stim_pred = stim_pred.detach().cpu()

        ### save
        with open(os.path.join(run_dir, f"config_{run_name}.json"), "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        torch.save({
            "run_config": run_config,
            "stim_pred": stim_pred,
        }, os.path.join(run_dir, f"ckpt_{run_name}.pt"), pickle_module=dill)
        plot_comparison(
            target=crop(stim, config["crop_win"]).cpu(),
            pred=crop(stim_pred, config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{run_name}.png"),
            show=False,
        )

    print(f"[INFO] Hyperparameter search finished. Best ({best['idx']}, val_loss={best['val_loss']}): {json.dumps(best['config'], indent=2, default=str)}")
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

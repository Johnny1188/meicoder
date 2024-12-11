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
from csng.models.ensemble import EnsembleInvEnc
# from csng.InvertedEncoder import InvertedEncoder, InvertedEncoderBrainreader
from csng.utils.mix import seed_all, plot_comparison, dict_to_str, slugify
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import find_best_ckpt, eval_decoder
from csng.losses import get_metrics
from csng.brainreader_mouse.encoder import get_encoder
from csng.brainreader_mouse.data import get_brainreader_mouse_dataloaders


lt.monkey_patch()
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")


##### global run config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": dict(),
    "crop_win": None,
}

### brainreader mouse data
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": "sequential",
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 16,
    "sessions": [6],
    "resize_stim_to": (36, 64),
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
        "encoder_paths": [
            os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
        ],
        "encoder_config": {
            "img_dims": (1, 36, 64),
            "stim_pred_init": "randn",
            "lr": 1000,
            "n_steps": 1000,
            "img_grad_gauss_blur_sigma": 1.5,
            "jitter": None,
            "mse_reduction": "per_sample_mean_sum",
            "device": config["device"],
        },
        "use_brainreader_encoder": True,
        "get_encoder_fn": get_encoder,
        "device": config["device"],
    },
    "sample_from": "val",
    "eval_on": "val",
    "loss_fns": get_metrics(crop_win=config["crop_win"], device=config["device"]),
    "save_dir": os.path.join(DATA_PATH_BRAINREADER, "models", "inverted_encoder"),
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
# ]
config_grid_search = None
config_grid_search = {
    "n_steps": [300, 1000, 2000],
    "lr": [100, 1000, 2000],
    "img_grad_gauss_blur_sigma": [1, 1.5, 2, 2.5],
    "jitter": [0],
}


if __name__ == "__main__":
    assert len(config["data"]["brainreader_mouse"]["sessions"]) == 1, "Only one session supported for now"
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    print(f"{DATA_PATH_BRAINREADER=}")
    seed_all(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["enc_inv"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["enc_inv"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dls = get_brainreader_mouse_dataloaders(config["data"]["brainreader_mouse"])["brainreader_mouse"]
    data_key = dls[config["enc_inv"]["sample_from"]].data_keys[0]
    datapoint = next(iter(dls[config["enc_inv"]["sample_from"]].dataloaders[0]))
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
        run_config["enc_inv"]["model"]["encoder_config"].update(config_update)
        run_name = f"{i}__{slugify(config_update)}"
        model = EnsembleInvEnc(**run_config["enc_inv"]["model"]).to(config["device"])

        ### eval
        if config["enc_inv"]["eval_on"] is not None:
            dls = get_brainreader_mouse_dataloaders(config["data"]["brainreader_mouse"])["brainreader_mouse"]
            val_losses = eval_decoder(
                model=model,
                dataloaders={"brainreader_mouse": dls[config["enc_inv"]["eval_on"]]},
                loss_fns={data_key: config["enc_inv"]["loss_fns"]},
                calc_fid=config["enc_inv"]["find_best_ckpt_according_to"] == "FID",
                crop_wins={data_key: config["crop_win"]},
                max_batches=config["enc_inv"]["max_batches"],
            )[data_key]

            ### update best
            val_loss = val_losses[config["enc_inv"]["find_best_ckpt_according_to"]]
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
            data_key=data_key,
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

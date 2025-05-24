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
from csng.data import get_dataloaders
from csng.utils.mix import seed_all, plot_comparison, dict_to_str, slugify, check_if_data_zscored
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import find_best_ckpt, eval_decoder
from csng.losses import get_metrics
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.mouse_v1.encoder import get_encoder as get_encoder_mouse_v1
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1

get_encoder_fns = {
    "brainreader_mouse": get_encoder_brainreader,
    "mouse_v1": get_encoder_mouse_v1,
    "cat_v1": get_encoder_cat_v1,
}

lt.monkey_patch()
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")


##### global run config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {"mixing_strategy": "sequential"},
    "crop_win": None,
    "data_name": "brainreader_mouse",
}

### data config
if config["data_name"] == "brainreader_mouse":
    config["data"]["brainreader_mouse"] = {
        "device": config["device"],
        "mixing_strategy": "sequential",
        "max_batches": None,
        "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
        "batch_size": 32,
        "sessions": [6],
        "resize_stim_to": (36, 64),
        "normalize_stim": True,
        "normalize_resp": False,
        "div_resp_by_std": True,
        "clamp_neg_resp": False,
        "additional_keys": None,
        "avg_test_resp": True,
    }
    config["crop_win"] = None
elif config["data_name"] == "cat_v1":
    config["data"]["cat_v1"] = {
        "dataset_config": {
            "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
            "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
            "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
            "image_size": [50, 50],
            "crop": False,
            "batch_size": 32,
            "stim_keys": ("stim",),
            "resp_keys": ("exc_resp", "inh_resp"),
            "return_coords": True,
            "return_ori": False,
            "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
            "cached": False,
            "stim_normalize_mean": 46.143,
            "stim_normalize_std": 24.960,
            "resp_normalize_mean": None, # don't center responses
            "resp_normalize_std": torch.load(
                os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
            ),
            "clamp_neg_resp": False,
            "neuron_idxs": None,
            # "neuron_idxs": np.random.default_rng(seed=config["seed"]).choice(46875, size=5000, replace=False),
        },
    }
    config["crop_win"] = (20, 20)
elif config["data_name"] == "mouse_v1":
    config["data"]["mouse_v1"] = {
        "dataset_fn": "sensorium.datasets.static_loaders",
        "dataset_config": {
            "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
                os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-1
                # os.path.join(DATA_PATH_MOUSE_V1, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-2
                # os.path.join(DATA_PATH_MOUSE_V1, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-3
                # os.path.join(DATA_PATH_MOUSE_V1, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-4
                # os.path.join(DATA_PATH_MOUSE_V1, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-5
            ],
            "normalize": True,
            "z_score_responses": False,
            "scale": 0.25, # 256x144 -> 64x36
            "include_behavior": False,
            "add_behavior_as_channels": False,
            "include_eye_position": True,
            "exclude": None,
            "file_tree": True,
            "cuda": "cuda" in config["device"],
            "batch_size": 32,
            "drop_last": True,
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
    config["crop_win"] = (22, 36)



### encoder inversion config
config["enc_inv"] = {
    "model": {
        "encoder_paths": [
            # os.path.join(DATA_PATH, "models", "encoders", "encoder_ball.pt"),
            os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
            # os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
            # os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
            # os.path.join(DATA_PATH, "models", "encoders", "encoder_c_5000neurons.pt"),
        ],
        "encoder_config": {
            "img_dims": (1, 36, 64),
            # "img_dims": (1, 50, 50),
            # "stim_pred_init": "randn",
            "stim_pred_init": "zeros",
            # "lr": 2000,
            "opter_config": {"lr": 10},
            # "n_steps": 1000,
            "n_steps": 100,
            # "img_grad_gauss_blur_sigma": 1.5,
            "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.5},
            # "jitter": None,
            # "mse_reduction": "per_sample_mean_sum",
            "device": config["device"],
        },
        "use_brainreader_encoder": False,
        "get_encoder_fn": get_encoder_fns[config["data_name"]],
        "device": config["device"],
    },
    "sample_from": "val",
    "eval_on": "val",
    "loss_fns": get_metrics(inp_zscored=check_if_data_zscored(cfg=config), crop_win=config["crop_win"], device=config["device"]),
    "save_dir": os.path.join(DATA_PATH, "models", "inverted_encoder"),
    "find_best_ckpt_according_to": "Alex(5) Loss",
    "max_batches": None,
}

### hyperparam runs config - either manually selected or grid search
config_updates = [dict()]
config_grid_search = None
config_grid_search = {
    ### brainreader-style
    # "n_steps": [100, 500, 2000],
    # "lr": [500, 2000, 4000],
    # "img_grad_gauss_blur_sigma": [1, 1.5, 2, 2.5],
    # "jitter": [0],

    ### not brainreader-style
    "n_steps": [100, 200, 500, 1000],
    "opter_config": [{"lr": lr} for lr in [5, 10, 20, 50]],
    "img_grad_gauss_blur_config": [{"kernel_size": 13, "sigma": s} for s in [1, 1.5, 2, 2.5]],
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    print(f"{DATA_PATH_BRAINREADER=}")
    print(f"{DATA_PATH_CAT_V1=}")
    print(f"{DATA_PATH_MOUSE_V1=}")
    seed_all(config["seed"])

    ### modify config
    if "cat_v1" in config["data_name"]:
        enc_ckpt_cat_v1_neuron_idxs = torch.load(config["enc_inv"]["model"]["encoder_paths"][0])["config"]["data"]["cat_v1"]["dataset_config"]["neuron_idxs"]
        config["data"]["cat_v1"]["dataset_config"]["neuron_idxs"] = enc_ckpt_cat_v1_neuron_idxs
        print(f"[INFO] Using {len(enc_ckpt_cat_v1_neuron_idxs) if enc_ckpt_cat_v1_neuron_idxs is not None else 'all'} neurons from encoder checkpoint.")

    ### prepares dirs
    run_dir = os.path.join(config["enc_inv"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["enc_inv"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dls = get_dataloaders(config=config)[0][config["enc_inv"]["sample_from"]][config["data_name"]]
    assert len(dls.data_keys) == 1, "Only single data key supported for now."
    data_key = dls.data_keys[0]
    datapoint = next(iter(dls.dataloaders[0]))
    stim, resp = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"])

    ### prepare config_updates
    if config_grid_search is not None:
        keys, vals = zip(*config_grid_search.items())
        config_updates.extend([dict(zip(keys, v)) for v in itertools.product(*vals)])
    print(f"[INFO] Config updates to try:\n ", "\n  ".join([dict_to_str(config_update) for config_update in config_updates]))

    ### run
    best = {"config": None, "val_loss": np.inf, "idx": None, "run_name": None}
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
            dls = get_dataloaders(config=config)[0][config["enc_inv"]["eval_on"]]
            val_loss = eval_decoder(
                model=model,
                dataloaders=dls,
                loss_fns={data_key: config["enc_inv"]["loss_fns"]},
                crop_wins={data_key: config["crop_win"]},
                max_batches=config["enc_inv"]["max_batches"],
                eval_every_n_samples=None,
            )[data_key][config["enc_inv"]["find_best_ckpt_according_to"]]

            ### update best
            print(f"  val_loss={val_loss:.3f}", end="")
            if val_loss < best["val_loss"]:
                print(f" >>> new best", end="")
                best["val_loss"] = val_loss
                best["config"] = run_config
                best["run_name"] = run_name
                best["idx"] = i
            print("")
            print(f"   {slugify(config_update)}")

        ### save
        with open(os.path.join(run_dir, f"config_{run_name}.json"), "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        stim_pred = model(resp=resp, data_key=data_key).detach().cpu()
        torch.save({
            "run_config": run_config,
            "stim_pred": stim_pred,
        }, os.path.join(run_dir, f"ckpt_{run_name}.pt"), pickle_module=dill)

        ### plot sample
        plot_comparison(
            target=crop(stim, config["crop_win"]).cpu(),
            pred=crop(stim_pred, config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{run_name}.png"),
            show=False,
        )

    print(
        f"[INFO] Hyperparameter search finished.\n"
        f"  Best ({best['idx']}, val_loss={best['val_loss']}):\n"
        f"  Full config: {json.dumps(best['config'], indent=2, default=str)}"
        f"  Run name: {best['run_name']}\n"
        f"  Run dir: {run_dir}"
    )
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

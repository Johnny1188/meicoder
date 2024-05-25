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
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import lovely_tensors as lt
import dill
import itertools

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.utils import crop, plot_comparison, dict_to_str, standardize, normalize, count_parameters
from csng.comparison import get_metrics
from encoder import get_encoder
from data_utils import get_mouse_v1_data
from comparison_utils import eval_decoder

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


### prepare config
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "crop_win": (22, 36),
}

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

### Encoder inversion config
config["enc_inv"] = {
    "model": {
        "encoder": get_encoder(
            ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
            eval_mode=True,
            device=config["device"],
        ),
        "img_dims": (1, 36, 64),
        "stim_pred_init": "zeros",
        "opter_cls": torch.optim.SGD,
        "opter_config": {"lr": 50},
        "n_steps": 500,
        "resp_loss_fn": lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
        "stim_loss_fn": None, # set below
        "img_gauss_blur_config": None,
        "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2},
        "device": config["device"],
    },
    "loss_fns": get_metrics(config=config),
    "save_dir": os.path.join(DATA_PATH, "models", "inverted_encoder"),
    "find_best_ckpt_according_to": "SSIML-PL",
    # "find_best_ckpt_according_to": "FID",
    "max_batches": None,
}
if config["enc_inv"]["find_best_ckpt_according_to"] != "FID":
    config["enc_inv"]["model"]["stim_loss_fn"] = config["enc_inv"]["loss_fns"][config["enc_inv"]["find_best_ckpt_according_to"]]
else:
    config["enc_inv"]["model"]["stim_loss_fn"] = config["enc_inv"]["loss_fns"]["SSIML-PL"]

### hyperparam runs config - either manually selected or grid search
config_updates = []
config_grid_search = {
    "n_steps": [100, 200, 300, 500, 1000],
    "opter_config": [{"lr": 50}, {"lr": 150}, {"lr": 500}, {"lr": 1000}],
    "img_grad_gauss_blur_config": [{"kernel_size": 13, "sigma": 1}, {"kernel_size": 13, "sigma": 1.5}, {"kernel_size": 13, "sigma": 2}, {"kernel_size": 13, "sigma": 2.5}],
}


def plot_decoding_history(decoding_history, save_to=None, show=True):
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.plot(decoding_history["resp_loss"])
    ax.set_title("resp_loss")

    ax = fig.add_subplot(122)
    ax.plot(decoding_history["stim_loss"])
    ax.set_title("stim_loss")

    if show:
        plt.show()

    if save_to is not None:
        fig.savefig(save_to)


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["enc_inv"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["enc_inv"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dataloaders, neuron_coords = get_mouse_v1_data(config["data"])
    sample_data_key = dataloaders["mouse_v1"]["val"].data_keys[0]
    datapoint = next(iter(dataloaders["mouse_v1"]["val"].dataloaders[0]))
    stim, resp, pupil_center = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"]), datapoint.pupil_center.to(config["device"])

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
        model = InvertedEncoder(**run_config["enc_inv"]["model"]).to(config["device"])

        ### eval on validation dataset
        dls, neuron_coords = get_mouse_v1_data(config["data"])
        val_losses = eval_decoder(
            model=model,
            dataloader=dls["mouse_v1"]["val"],
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
        print(f"   {dict_to_str(config_update)}")

        ### plot sample
        stim_pred, _, decoding_history = model(
            resp_target=resp,
            stim_target=stim,
            additional_encoder_inp={
                "data_key": sample_data_key,
                "pupil_center": pupil_center,
            },
        )
        stim_pred = stim_pred.detach().cpu()
        stim_pred_best = decoding_history["best"]["stim_pred"].detach().cpu()

        ### save
        with open(os.path.join(run_dir, f"config_{i}.json"), "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        torch.save({
            "run_config": run_config,
            "stim_pred": stim_pred,
            "stim_pred_best": stim_pred_best,
            "decoding_history": decoding_history,
        }, os.path.join(run_dir, f"ckpt_{i}.pt"), pickle_module=dill)
        plot_comparison(
            target=crop(stim[:8], config["crop_win"]).cpu(),
            pred=crop(stim_pred[:8], config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{i}.png"),
            show=False,
        )
        plot_decoding_history(
            decoding_history=decoding_history,
            save_to=os.path.join(run_dir, f"decoding_history_{i}.png"),
            show=False,
        )

    print(f"[INFO] Hyperparameter search finished. Best ({best['idx']}, val_loss={best['val_loss']}):")
    json.dumps(best["config"], indent=2, default=str)
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

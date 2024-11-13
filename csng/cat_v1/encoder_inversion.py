import os
import itertools
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

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.utils import crop, plot_comparison, dict_to_str, standardize, normalize, get_mean_and_std, count_parameters
from csng.cat_v1.encoder import get_encoder
from csng.cat_v1.comparison_utils import eval_decoder, get_metrics, get_dataloaders

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")


### prepare config
config = {
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
    },
    "crop_win": (20, 20),
    "only_cat_v1_eval": True,
    "device": os.environ["DEVICE"],
    "seed": 0,
    "wandb": None,
}
config["data"]["cat_v1"] = {
    "train_path": os.path.join(DATA_PATH, "datasets", "train"),
    "val_path": os.path.join(DATA_PATH, "datasets", "val"),
    "test_path": os.path.join(DATA_PATH, "datasets", "test"),
    "image_size": [50, 50],
    "crop": False,
    "batch_size": 128,
    "stim_keys": ("stim",),
    "resp_keys": ("exc_resp", "inh_resp"),
    "return_coords": True,
    "return_ori": False,
    "coords_ori_filepath": os.path.join(DATA_PATH, "pos_and_ori.pkl"),
    "cached": False,
    "stim_normalize_mean": 46.143,
    "stim_normalize_std": 20.420,
    "resp_normalize_mean": None,
    "resp_normalize_std": None,
    # "resp_normalize_mean": torch.load(
    #     os.path.join(DATA_PATH, "responses_mean.pt")
    # ),
    # "resp_normalize_std": torch.load(
    #     os.path.join(DATA_PATH, "responses_std.pt")
    # ),
    # "clamp_neg_resp": True,
    "clamp_neg_resp": False,
}


# def resp_loss_fn(resp_pred, resp_target):
#     idxs = np.random.choice(resp_pred.size(1), size=5000, replace=False)
#     return F.mse_loss(resp_pred[:,idxs], resp_target[:,idxs], reduction="none").mean(-1).sum()

### Encoder inversion config
config["enc_inv"] = {
    "model": {
        "encoder": get_encoder(
            device=config["device"],
            eval_mode=True,
            ckpt_path=os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter_mean_activity.pth"),
        ),
        "img_dims": (1, 50, 50),
        "stim_pred_init": "zeros",
        "opter_cls": torch.optim.SGD,
        "opter_config": {"lr": 20},
        "n_steps": 200,
        "resp_loss_fn": lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
        # "resp_loss_fn": resp_loss_fn,
        "stim_loss_fn": None, # set below
        "img_gauss_blur_config": None,
        "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.5},
        "device": config["device"],
    },
    "loss_fns": get_metrics(config=config),
    "save_dir": os.path.join(DATA_PATH, "models", "inverted_encoder"),
    "find_best_ckpt_according_to": "SSIML-PL",
    # "find_best_ckpt_according_to": "SSIML",
    # "max_batches": None,
    "max_batches": 10,
}
config["enc_inv"]["model"]["stim_loss_fn"] = config["enc_inv"]["loss_fns"][config["enc_inv"]["find_best_ckpt_according_to"]]



### hyperparam runs config
# config_updates = [
#     {},
#     # {"n_steps": 1000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.0}},
#     # {"n_steps": 5000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.0}},
    
#     # {"n_steps": 3000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.5}},
#     # {"n_steps": 1000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.5}},
#     # {"n_steps": 500, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.5}},
    
#     # {"n_steps": 3000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.5}},
#     # {"n_steps": 1000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.5}},
#     # {"n_steps": 500, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 2.5}},

#     # {"n_steps": 3000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.}},
#     # {"n_steps": 1000, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.}},
#     # {"n_steps": 500, "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.}},
# ]
config_updates = None
config_grid_search = {
    "n_steps": [100, 200, 500, 1000],
    "opter_config": [{"lr": 10}, {"lr": 20}, {"lr": 50}],
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
    print(f"[INFO] ... Running on {config['device']} ...")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["enc_inv"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["enc_inv"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Config:\n{json.dumps(config['enc_inv'], indent=2, default=str)}")
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    ### sample data
    _, _, test_dl = get_dataloaders(config=config)
    sample_data_key = "cat_v1"
    datapoint = next(iter(test_dl))
    stim, resp, neuron_coords = datapoint[sample_data_key][0].to(config["device"]), datapoint[sample_data_key][1].to(config["device"]), datapoint[sample_data_key][2].float().to(config["device"])
    stim, resp, neuron_coords = stim[:7], resp[:7], neuron_coords[:7]

    ### prepare config_updates
    if config_updates is None:
        keys, vals = zip(*config_grid_search.items())
        config_updates = [dict(zip(keys, v)) for v in itertools.product(*vals)]
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
        _, val_dl, _ = get_dataloaders(config=config)
        val_losses = eval_decoder(
            model=model,
            dataloader=val_dl,
            loss_fns=config["enc_inv"]["loss_fns"],
            config=config,
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
            },
            # ckpt_config={
            #     "ckpt_dir": os.path.join(run_dir, "ckpt"),
            #     "ckpt_freq": 3000,
            #     "plot_fn": lambda target, pred, save_to: plot_comparison(
            #         target=crop(target[:8].cpu(), config["crop_win"]).cpu(),
            #         pred=crop(pred[:8].detach().cpu(), config["crop_win"]).cpu(),
            #         save_to=save_to,
            #         show=False,
            #     ) and plt.close(),
            # },
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
        plot_comparison(
            target=stim[:8].cpu(),
            pred=stim_pred[:8].cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_no_crop_{i}.png"),
            show=False,
        )
        plot_comparison(
            target=crop(stim[:8], config["crop_win"]).cpu(),
            pred=crop(stim_pred_best[:8], config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_best_{i}.png"),
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

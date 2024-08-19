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
import featurevis
from featurevis import ops
from featurevis import utils as fvutils

import csng
from csng.utils import crop, plot_comparison, dict_to_str, standardize, normalize, count_parameters, slugify
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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "crop_win": (36, 64),
}

### data config
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH, "data"),
    "batch_size": 1,
    "sessions": list(range(1, 2)),
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}


### Encoder inversion config
config["enc_inv"] = {
    "model": {
        "encoder": get_encoder(
            ckpt_path=os.path.join(DATA_PATH, "models", "encoder_mall.pth"),
            eval_mode=True,
            device=config["device"],
        ),
        "img_dims": (1, 36, 64),
        "stim_pred_init": "randn",
        "lr": 100,
        "n_steps": 1000,
        "img_grad_gauss_blur_sigma": 2,
        "jitter": 0,
        "mse_reduction": "per_sample_mean_sum",
        "device": config["device"],
    },
    "loss_fns": get_metrics(crop_win=config["crop_win"], device=config["device"]),
    "save_dir": os.path.join(DATA_PATH, "models", "inverted_encoder"),
    # "find_best_ckpt_according_to": "SSIML-PL",
    "find_best_ckpt_according_to": "FID",
    "max_batches": None,
}

### hyperparam runs config - either manually selected or grid search
config_updates = []
config_grid_search = {
    "n_steps": [250, 500, 750],
    "lr": [100, 250, 500, 1000],
    "img_grad_gauss_blur_sigma": [1.5, 2, 2.5],
    "jitter": [0],
}



class InvertedEncoderBrainreader(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 36, 64),
        stim_pred_init="randn",
        lr=100,
        n_steps=500,
        img_grad_gauss_blur_sigma=0,
        jitter=None,
        mse_reduction="per_sample_mean_sum",
        device="cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.training = False
        self.encoder.eval()

        self.mse_reduction = mse_reduction
        self.stim_pred_init = stim_pred_init
        self.img_dims = img_dims
        self.lr = lr
        self.n_steps = n_steps

        self.img_grad_gauss_blur_sigma = img_grad_gauss_blur_sigma
        self.jitter = jitter

        self.device = device

    def _init_x_hat(self, B):
        ### init decoded img
        if self.stim_pred_init == "zeros":
            x_hat = torch.zeros((B, *self.img_dims), requires_grad=True, device=self.device)
        elif self.stim_pred_init == "rand":
            x_hat = torch.rand((B, *self.img_dims), requires_grad=True, device=self.device)
        elif self.stim_pred_init == "randn":
            x_hat = torch.randn((B, *self.img_dims), requires_grad=True, device=self.device)
        else:
            raise ValueError(f"Unknown stim_pred_init: {self.stim_pred_init}")
        return x_hat

    def forward(self, resp_target, stim_target=None, additional_encoder_inp=None):
        assert resp_target.ndim > 1, "resp_target should be at least 2d (batch_dim, neurons_dim)"

        ### run separately for each image
        recons = []
        # Set up initial image
        torch.manual_seed(0)
        initial_image = self._init_x_hat(resp_target.shape[0])

        # Set up optimization function
        neural_resp = torch.as_tensor(resp_target, dtype=torch.float32, device=self.device)
        similarity = fvutils.Compose([ops.MSE(neural_resp, reduction=self.mse_reduction), ops.MultiplyBy(-1)])
        encoder_pred = lambda x: self.encoder(x, data_key=additional_encoder_inp["data_key"])
        obj_function = fvutils.Compose([encoder_pred, similarity])

        # Optimize
        jitter = ops.Jitter(self.jitter) if self.jitter is not None and self.jitter != 0 else None
        blur = (ops.GaussianBlur(self.img_grad_gauss_blur_sigma)
                if self.img_grad_gauss_blur_sigma != 0 else None)
        recon, fevals, _ = featurevis.gradient_ascent(
            obj_function,
            initial_image,
            step_size=self.lr,
            num_iterations=self.n_steps,
            transform=jitter,
            gradient_f=blur,
            regularization=None,
            post_update=ops.ChangeStd(1),
            print_iters=None,
        )
        recons.append(recon.detach())

        return torch.cat(recons, dim=0), None, None



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
    dls = get_brainreader_data(config=config["data"]["brainreader_mouse"])
    sample_data_key = dls["brainreader_mouse"]["val"].data_keys[0]
    datapoint = next(iter(dls["brainreader_mouse"]["val"].dataloaders[0]))
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
        model = InvertedEncoderBrainreader(**run_config["enc_inv"]["model"]).to(config["device"])

        ### eval on validation dataset
        dls = get_brainreader_data(config["data"]["brainreader_mouse"])
        val_losses = eval_decoder(
            model=model,
            dataloaders={"brainreader_mouse": dls["brainreader_mouse"]["val"]},
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
        stim_pred, _, _ = model(
            resp_target=resp,
            additional_encoder_inp={
                "data_key": sample_data_key,
            },
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
            target=crop(stim[:8], config["crop_win"]).cpu(),
            pred=crop(stim_pred[:8], config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{run_name}.png"),
            show=False,
        )

    print(f"[INFO] Hyperparameter search finished. Best ({best['idx']}, val_loss={best['val_loss']}): {json.dumps(best['config'], indent=2, default=str)}")
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

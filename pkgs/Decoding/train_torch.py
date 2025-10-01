import numpy as np
import torch
import dill
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm # For a nice progress bar

import os
import json
import wandb
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()

from csng.data import get_dataloaders
from csng.utils.mix import seed_all, check_if_data_zscored
from csng.utils.data import crop
from csng.losses import get_metrics
from cae.model import CAEDecoder

DATA_PATH_CAE = os.path.join(os.environ["DATA_PATH"], "cae")
DATA_PATH_BRAINREADER = os.path.join(os.environ["DATA_PATH"], "brainreader")
DATA_PATH_CAT_V1 = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


### setup config
cfg = {
    "device": os.environ.get("DEVICE", "cpu"),
    "seed": 0,
    "run_name": datetime.now().strftime("%d-%m-%Y_%H-%M"),
    "data": {
        "data_name": "cat_v1",
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
        "img_transforms": {
            "brainreader_mouse": lambda x: crop(x, (36, 64)),
            "mouse_v1": lambda x: crop(x, (22, 36)),
            "cat_v1": lambda x: crop(x, (20, 20)),
            "allen": lambda x: x,
        },
    },
    # "wandb": None,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "cae",
    },
}

### data
if cfg["data"]["data_name"] == "brainreader_mouse":
    cfg["data"]["brainreader_mouse"] = {
        "device": cfg["device"],
        "mixing_strategy": cfg["data"]["mixing_strategy"],
        "max_batches": None,
        "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
        "batch_size": 32,
        # "sessions": list(range(1, 23)),
        "sessions": [6],
        "resize_stim_to": (36, 64),
        "normalize_stim": True,
        "normalize_resp": False,
        "div_resp_by_std": True,
        "clamp_neg_resp": False,
        "additional_keys": None,
        "avg_test_resp": True,
    }
elif cfg["data"]["data_name"] == "mouse_v1":
    cfg["data"]["mouse_v1"] = {
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
            "cuda": "cuda" in cfg["device"],
            "batch_size": 32,
            "drop_last": True,
            "use_cache": False,
        },
        "skip_train": False,
        "skip_val": False,
        "skip_test": False,
        "normalize_neuron_coords": True,
        "average_test_multitrial": True,
        "save_test_multitrial": True,
        "test_batch_size": 7,
        "device": cfg["device"],
    }
elif cfg["data"]["data_name"] == "cat_v1":
    cfg["data"]["cat_v1"] = {
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
            # "neuron_idxs": np.random.default_rng(seed=cfg["seed"]).choice(46875, size=5000, replace=False),
        },
    }
elif cfg["data"]["data_name"] == "allen":
    cfg["data"]["allen"] = {
        "device": cfg["device"],
        "val_split_seed": cfg["seed"],
        "mixing_strategy": "sequential",
        "batch_size": 16,
        "val_split_frac": 0.2,
    }

### model config
cfg["model"] = {
    "kwargs": {
        "ncell": (tmp_batch := next(iter(get_dataloaders(config=cfg)[0]["train"][cfg["data"]["data_name"]]))[0])["resp"].shape[1],
        "intermediate_shape": (
            1,
            cfg["data"]["img_transforms"][cfg["data"]["data_name"]](tmp_batch["stim"]).size(-2) // 2,
            cfg["data"]["img_transforms"][cfg["data"]["data_name"]](tmp_batch["stim"]).size(-1) // 2
        ) if cfg["data"]["data_name"] != "allen" else (1, 64, 64),  # Shape after DenseDecoder output
        "size": "small" if cfg["data"]["data_name"] != "allen" else "large",  # CAE size
    },
    "opter_kwargs": {
        "lr": 1e-3,
        "weight_decay": 0,
    },
    "epochs": 500,
    "save_dir": (save_dir := os.path.join(DATA_PATH_CAE, "runs", cfg["run_name"])),
    "save_to": os.path.join(save_dir, f"model.pt"),
    "save_predictions_to": os.path.join(save_dir, f"predictions.npy"),
}


# --- 3. Main Execution Block ---
if __name__ == '__main__':
    print(f"... Running on {cfg['device']} ...")
    os.makedirs(cfg["model"]["save_dir"], exist_ok=True)

    assert (
        ("brainreader_mouse" not in cfg["data"] or cfg["data"]["brainreader_mouse"]["sessions"] == [6]) and
        ("mouse_v1" not in cfg["data"] or cfg["data"]["mouse_v1"]["dataset_config"]["paths"] == [os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip")])
    ), "Only B-6 and M-1 are supported now."
    img_transforms = cfg["data"]["img_transforms"][cfg["data"]["data_name"]]

    ### w&b
    wdb_run = None
    if cfg["wandb"] is not None:
        wdb_run = wandb.init(**cfg["wandb"], name=cfg["run_name"], config=cfg, save_code=True,
            tags=["CAE"] + list(cfg["data"].keys()), id=cfg["run_name"])
        wdb_run.log_code(".", include_fn=lambda path, root: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") or path.endswith(".yml"))

    # --- Model, Loss, and Optimizer ---
    seed_all(cfg["seed"])
    model = CAEDecoder(**cfg["model"]["kwargs"]).to(cfg["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), **cfg["model"]["opter_kwargs"])
    if wdb_run is not None:
        wdb_run.watch(model, log="all")
    print("Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    ### load checkpoints
    if cfg["model"].get("load_ckpt") is not None:
        print(f"[INFO] Loading checkpoint from {cfg['model']['load_ckpt']} ...")
        ckpt = torch.load(cfg['model']['load_ckpt'], pickle_module=dill)
        model.load_state_dict(ckpt["model"])
    else:
        print("[INFO] No checkpoint loaded.")

    # --- Training Loop ---
    print("Starting training...")
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(cfg["model"]["epochs"]):
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        dls, _ = get_dataloaders(config=cfg)
        train_dl, val_dl = dls["train"][cfg["data"]["data_name"]], dls["val"][cfg["data"]["data_name"]]
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['model']['epochs']}", leave=False)
        for batch in progress_bar:
            spikes = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
            images = img_transforms(torch.cat([dp["stim"] for dp in batch], dim=0).to(cfg["device"]))

            optimizer.zero_grad()
            outputs = img_transforms(model(spikes))

            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                spikes = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
                images = img_transforms(torch.cat([dp["stim"] for dp in batch], dim=0).to(cfg["device"]))
                outputs = img_transforms(model(spikes))

                loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)

        print(f"Epoch {epoch+1}/{cfg['model']['epochs']} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        if wdb_run is not None:
            wdb_run.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            })

        # --- Checkpointing ---
        if avg_val_loss < best["val_loss"]:
            best["val_loss"] = avg_val_loss
            best["epoch"] = epoch + 1
            save_best_to = os.path.join(
                cfg["model"]["save_dir"],
                "best_" + os.path.basename(cfg["model"]["save_to"])
            )
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, save_best_to, pickle_module=dill)
            print(f"New best model saved with val_loss: {best['val_loss']:.5f} at epoch {best['epoch']}")

        # Save model checkpoint after each epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, cfg["model"]["save_to"], pickle_module=dill)

    print("Training finished.")

    # --- Save Model Weights ---
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, cfg["model"]["save_to"], pickle_module=dill)
    print(f"Model weights saved to {cfg['model']['save_to']}")

    # --- Prediction ---
    if cfg["model"]["save_predictions_to"] is not None:
        print("Generating predictions on the test set...")
        dls, _ = get_dataloaders(config=cfg)
        model.eval()
        pred_cae_list = []

        with torch.no_grad():
            for batch in tqdm(dls["test"][cfg["data"]["data_name"]], desc="Generating predictions", leave=False):
                spikes = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
                pred_cae = img_transforms(model(spikes))
                pred_cae_list.append(pred_cae.cpu().numpy())

        # Concatenate all predictions into a single numpy array
        pred_cae_np = np.concatenate(pred_cae_list, axis=0)

        # --- Save Prediction Results ---
        np.save(cfg["model"]["save_predictions_to"], pred_cae_np)
        print(f"Prediction results saved to {cfg['model']['save_predictions_to']}")

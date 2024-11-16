import os
import numpy as np
import torch
import json
import dill
import wandb
from datetime import datetime
from collections import defaultdict

from csng.utils.mix import timeit
from csng.utils.data import crop
from csng.losses import SSIMLoss, Loss, FID
from csng.models.readins import MultiReadIn

### set paths
DATA_PATH = os.environ["DATA_PATH"]



def init_decoder(config):
    if config["decoder"]["load_ckpt"] != None:
        print(f"[INFO] Loading checkpoint from {config['decoder']['load_ckpt']['ckpt_path']}...")
        ckpt = torch.load(config["decoder"]["load_ckpt"]["ckpt_path"], map_location=config["device"], pickle_module=dill)

        if config["decoder"]["load_ckpt"]["load_only_core"]:
            print("[INFO] Loading only the core of the model (no history, no best ckpt)...")

            ### init decoder (load only the core)
            config["decoder"]["model"]["core_cls"] = ckpt["config"]["decoder"]["model"]["core_cls"]
            config["decoder"]["model"]["core_config"] = ckpt["config"]["decoder"]["model"]["core_config"]
            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict({k:v for k,v in ckpt["best"]["model"].items() if "readin" not in k}, strict=False)

            ### init the rest
            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
            history = {"train_loss": [], "val_loss": []}
            best = {"val_loss": np.inf, "epoch": 0, "model": None}
        else:
            print("[INFO] Continuing the training run (loading the current model, history, and overwriting the config)...")
            history, best, config["decoder"]["model"] = ckpt["history"], ckpt["best"], ckpt["config"]["decoder"]["model"]

            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict(ckpt["decoder"])

            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            opter.load_state_dict(ckpt["opter"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
    else:
        print("[INFO] Initializing the model from scratch...")
        ckpt = None
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])

        history = {"train_loss": [], "val_loss": []}
        best = {"val_loss": np.inf, "epoch": 0, "model": None}

    return config, decoder, opter, loss_fn, history, best, ckpt


def setup_run_dir(config, ckpt=None):
    if config["decoder"]["load_ckpt"] == None or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ### save run and config        
        if config["save_run"]:
            config["dir"] = os.path.join(DATA_PATH, "models", "cnn", config["run_name"])
            os.makedirs(config["dir"], exist_ok=True)
            with open(os.path.join(config["dir"], "config.json"), "w") as f:
                json.dump(config, f, indent=4, default=str)
            os.makedirs(os.path.join(config["dir"], "samples"), exist_ok=True)
            os.makedirs(os.path.join(config["dir"], "ckpt"), exist_ok=True)
            make_sample_path = lambda epoch, prefix: os.path.join(
                config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
            )
            print(f"Run name: {config['run_name']}\nRun dir: {config['dir']}")
        else:
            make_sample_path = lambda epoch, prefix: None
            print("[WARNING] Not saving the run and the config.")
    else:
        ### resume checkpointing
        assert ckpt is not None, "Checkpoint to resume from is not provided."
        config["run_name"] = ckpt["config"]["run_name"]
        config["dir"] = ckpt["config"]["dir"]
        make_sample_path = lambda epoch, prefix: os.path.join(
            config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
        )
        print(f"Checkpointing resumed - Run name: {config['run_name']}\nRun dir: {config['dir']}")
    
    return config, make_sample_path


def setup_wandb_run(config, decoder=None):
    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_wandb_id"] == None:
        if config["wandb"]:
            wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config,
                tags=[
                    config["decoder"]["model"]["core_cls"].__name__,
                    config["decoder"]["model"]["readins_config"][0]["layers"][0][0].__name__,
                ],
                notes=None,
                save_code=True,
            )
            if decoder:
                wdb_run.watch(decoder)
        else:
            print("[WARNING] Not using wandb.")
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must", save_code=True)

    wdb_run.log_code(
        ".",
        include_fn=lambda path, root: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") or path.endswith(".yml"),
    )

    return wdb_run


# @timeit
def train(model, dataloaders, opter, loss_fn, config, verbose=True):
    model.train()
    train_loss = 0
    n_batches = max(len(dl) for dl in dataloaders.values())

    ### run
    batch_idx = 0
    while len(dataloaders) > 0:
        ### next batch
        opter.zero_grad()
        loss, n_dps = 0, 0
        dl_ks = list(dataloaders.keys())
        for k in dl_ks:
            dl = dataloaders[k]
            try:
                b = next(dl)
            except StopIteration:
                del dataloaders[k]
                continue

            ### combine from all data keys
            for dp in b:
                ### get loss
                stim_pred = model(
                    dp["resp"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    pupil_center=dp["pupil_center"],
                )
                loss += loss_fn(stim_pred, dp["stim"], data_key=dp["data_key"], phase="train", neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"])
                model.set_additional_loss(
                    inp={
                        "resp": dp["resp"],
                        "stim": dp["stim"],
                        "neuron_coords": dp["neuron_coords"],
                        "pupil_center": dp["pupil_center"],
                        "data_key": dp["data_key"],
                    }, out={
                        "stim_pred": stim_pred,
                    },
                )
                loss += model.get_additional_loss(data_key=dp["data_key"])
                n_dps += 1

        ### update
        if n_dps > 0:
            loss /= n_dps
            loss.backward()
            opter.step()

        ### log
        loss = loss.item() if n_dps > 0 else 0
        train_loss += loss
        if verbose and batch_idx % 100 == 0:
            print(f"Training progress: [{batch_idx}/{n_batches} ({100. * batch_idx / n_batches:.0f}%)]"
                    f"  Loss: {loss:.6f}")
        batch_idx += 1

    train_loss /= n_batches
    return train_loss


# @timeit
def val(model, dataloaders, loss_fn, config):
    model.eval()
    val_loss = 0
    n_samples = 0

    ### is loss_fn FID?
    is_fid = False
    if type(loss_fn) == str and loss_fn.lower() == "fid":
        is_fid = True
        preds, targets = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for k, dl in dataloaders.items():
            for batch_idx, b in enumerate(dl):
                ### combine from all data keys
                for dp in b:
                    ### predict
                    stim_pred = model(dp["resp"], data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"])
                    
                    ### calc loss/FID
                    if is_fid:
                        preds[dp["data_key"]].append(crop(stim_pred, config["crop_wins"][dp["data_key"]]).cpu())
                        targets[dp["data_key"]].append(crop(dp["stim"], config["crop_wins"][dp["data_key"]]).cpu())
                    else:
                        val_loss += loss_fn(stim_pred, dp["stim"], phase="val", data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"]).item()
                        n_samples += dp["resp"].shape[0]

    ### finalize the val loss
    if is_fid:
        for data_key in preds.keys():
            fid = FID(inp_standardized=False, device="cpu")
            val_loss += fid(
                pred_imgs=torch.cat(preds[data_key], dim=0),
                gt_imgs=torch.cat(targets[data_key], dim=0)
            )
        val_loss /= len(preds.keys()) # average over data keys
    else:
        val_loss /= n_samples

    return val_loss

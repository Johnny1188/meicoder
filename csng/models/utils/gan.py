import os
import numpy as np
import dill
import json
from datetime import datetime
import wandb
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

import csng
from csng.utils.mix import timeit
from csng.utils.data import standardize, crop
from csng.utils.models import TransparentDataParallel
from csng.losses import Loss, FID
from csng.models.readins import (
    MultiReadIn,
    ConvReadIn,
    FCReadIn,
    MEIReadIn,
)

### set paths
DATA_PATH = os.environ["DATA_PATH"]


def init_decoder(config, merge_configs_fn=None):
    history = {"val_loss": []}
    best = {"val_loss": np.inf, "epoch": 0, "model": None}

    ### initialize decoder (and load ckpt if needed)
    if config["decoder"]["load_ckpt"] != None:
        print(f"[INFO] Loading checkpoint from {config['decoder']['load_ckpt']['ckpt_path']}...")
        ckpt = torch.load(config["decoder"]["load_ckpt"]["ckpt_path"], map_location=config["device"], pickle_module=dill)
        ckpt_cfg = ckpt["config"]
        if merge_configs_fn is not None:
            config, ckpt_cfg = merge_configs_fn(config, ckpt_cfg)

        ### load decoder
        config["decoder"]["model"] = ckpt_cfg["decoder"]["model"]
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        decoder.load_from_ckpt(ckpt=ckpt, load_best=config["decoder"]["load_ckpt"]["load_best"],
            load_only_core=config["decoder"]["load_ckpt"]["load_only_core"], strict=config["decoder"]["load_ckpt"]["load_only_core"] is False)

        ### init optimizers (and load their states)
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])
        if config["decoder"]["load_ckpt"]["load_opter_state"]:
            if config["decoder"]["load_ckpt"]["load_best"]:
                core_state_dict = {".".join(k.split(".")[1:]):v for k,v in ckpt["best"]["decoder"].items() if "G" in k or "D" in k}
            else:
                core_state_dict = {".".join(k.split(".")[1:]):v for k,v in ckpt["decoder"].items() if "G" in k or "D" in k}
            decoder.core.G_optim.load_state_dict(core_state_dict["G_optim"])
            decoder.core.D_optim.load_state_dict(core_state_dict["D_optim"])

        ### history and best tracking
        if config["decoder"]["load_ckpt"]["load_history"]:
            history = ckpt["history"]
        if not config["decoder"]["load_ckpt"]["reset_best"]:
            best = ckpt["best"]
    else:
        print("[INFO] Initializing the model from scratch...")
        ckpt = None
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])

    loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])

    ### data parallelism
    # decoder = TransparentDataParallel(decoder).to(config["device"])

    return config, decoder, loss_fn, history, best, ckpt


def setup_run_dir(config, ckpt):
    if config["decoder"]["load_ckpt"] == None or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["save_run"]:
            ### save config
            config["dir"] = os.path.join(DATA_PATH, "models", "gan", config["run_name"])
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
            wdb_run = None
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must", save_code=True)

    if wdb_run:
        wdb_run.log_code(
            ".",
            include_fn=lambda path, root: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") or path.endswith(".yml"),
        )

    return wdb_run


def get_training_losses(model, resp, stim, data_key, neuron_coords, loss_fn, config, crop_win=None):
    ### get the stimulus reconstruction from the generator
    stim_pred = model(
        resp,
        data_key=data_key,
        neuron_coords=neuron_coords,
    )

    ### get discriminator loss
    # real stim (add noise to labels (uniform distribution between 0.xx and 1))
    real_stim_pred = model.core.D(crop(stim, config["crop_win"] if crop_win is None else crop_win), data_key=data_key)
    noisy_real_stim_labels = (
        1. - config["decoder"]["D_real_stim_labels_noise"] 
        + torch.rand_like(real_stim_pred) * config["decoder"]["D_real_stim_labels_noise"]
    )
    real_stim_loss = torch.mean((real_stim_pred - noisy_real_stim_labels)**2) * config["decoder"]["D_real_loss_mul"]
    # fake stim (add noise to labels (uniform distribution between 0 and noise level))
    fake_stim_pred = model.core.D(crop(stim_pred.detach(), config["crop_win"] if crop_win is None else crop_win), data_key=data_key)
    noisy_fake_stim_labels = torch.rand_like(fake_stim_pred) * config["decoder"]["D_fake_stim_labels_noise"]
    fake_stim_loss = torch.mean((fake_stim_pred - noisy_fake_stim_labels)**2) * config["decoder"]["D_fake_loss_mul"]
    D_loss = real_stim_loss + fake_stim_loss

    ### get generator loss
    # fooling the discriminator
    fake_stim_pred_for_G = model.core.D(crop(stim_pred, config["crop_win"] if crop_win is None else crop_win), data_key=data_key)
    G_loss_adv = torch.mean((fake_stim_pred_for_G - 1.)**2) * config["decoder"]["G_adv_loss_mul"]
    # reconstruction quality
    G_loss_stim = config["decoder"]["G_stim_loss_mul"] * loss_fn(stim_pred, stim, data_key=data_key, phase="train", neuron_coords=neuron_coords)
    G_loss = G_loss_adv + G_loss_stim

    return G_loss, G_loss_stim, G_loss_adv, D_loss, real_stim_loss, fake_stim_loss, stim_pred


def update_G(model, G_loss, config, data_keys=None):
    model.core.G_optim.zero_grad()

    ### regularization
    if config["decoder"]["G_reg"]["l2"] > 0:
        G_loss += sum(p.pow(2).sum() for n,p in model.core.G.named_parameters() \
            if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l2"]
        G_loss += sum(p.pow(2).sum() for n,p in model.readins.named_parameters() \
            if p.requires_grad and "bias" not in n and (data_keys is None or sum(f".{dk}." in n for dk in data_keys) == 1)) * config["decoder"]["G_reg"]["l2"]
    if config["decoder"]["G_reg"]["l1"] > 0:
        G_loss += sum(p.abs().sum() for n,p in model.core.G.named_parameters() \
            if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l1"]
        G_loss += sum(p.abs().sum() for n,p in model.readins.named_parameters() \
            if p.requires_grad and "bias" not in n and (data_keys is None or sum(f".{dk}." in n for dk in data_keys) == 1)) * config["decoder"]["G_reg"]["l1"]
    G_loss.backward()

    ### clip gradients
    for n, p in model.core.G.named_parameters():
        if p.grad != None:
            p.grad.data.clamp_(-1., 1.)

    ### log
    G_mean_abs_grad_first_layer = torch.mean(torch.abs(model.core.G.layers[0].weight.grad)).item()
    G_mean_abs_grad_last_layer = torch.mean(torch.abs(model.core.G.layers[-2].weight.grad)).item()

    ### step
    model.core.G_optim.step()

    return G_loss, G_mean_abs_grad_first_layer, G_mean_abs_grad_last_layer


def update_D(model, D_loss, config, data_keys=None):
    model.core.D_optim.zero_grad()

    ### regularization
    if config["decoder"]["D_reg"]["l2"] > 0:
        D_loss += sum(p.pow(2).sum() for n,p in model.core.D.named_parameters() \
            if p.requires_grad and "bias" not in n and ("head" not in n or sum(f".{dk}." in n for dk in data_keys) == 1)) * config["decoder"]["D_reg"]["l2"]
    if config["decoder"]["D_reg"]["l1"] > 0:
        D_loss += sum(p.abs().sum() for n,p in model.core.D.named_parameters() \
            if p.requires_grad and "bias" not in n and ("head" not in n or sum(f".{dk}." in n for dk in data_keys) == 1)) * config["decoder"]["D_reg"]["l1"]
    D_loss.backward()

    ### clip gradients
    for n, p in model.core.D.named_parameters():
        if p.grad != None:
            p.grad.data.clamp_(-1., 1.)

    ### log
    D_mean_abs_grad_first_layer = torch.mean(torch.abs(model.core.D.layers[0].weight.grad)).item()
    if model.core.D.head is None:
        D_mean_abs_grad_last_layer = torch.mean(torch.abs(model.core.D.layers[-2].weight.grad)).item()
    else:
        D_mean_abs_grad_last_layer, div_by = 0, 0
        for n, p in model.core.D.head.named_parameters():
            if p.grad is None:
                continue
            if "weight" in n:
                D_mean_abs_grad_last_layer += p.grad.abs().mean().item()
                div_by += 1
        D_mean_abs_grad_last_layer /= max(div_by, 1)

    ### step
    model.core.D_optim.step()

    return D_loss, D_mean_abs_grad_first_layer, D_mean_abs_grad_last_layer


# @timeit
def train(model, dataloaders, loss_fn, config, history, log_freq=100, wdb_run=None, wdb_commit=True):
    for k in ("G_mean_abs_grad_first_layer", "G_mean_abs_grad_last_layer", "D_mean_abs_grad_first_layer", "D_mean_abs_grad_last_layer",
        "D_loss", "G_loss", "G_loss_stim", "G_loss_adv", "D_loss_real", "D_loss_fake"):
        if k not in history.keys(): history[k] = []

    model.train()
    n_batches = max(len(dl) for dl in dataloaders.values())

    ### run
    batch_idx = 0
    while len(dataloaders) > 0:
        ### next batch
        n_dps, seen_data_keys = 0, set()
        D_loss, D_real_stim_loss, D_fake_stim_loss, G_loss, G_loss_stim, G_loss_adv = 0, 0, 0, 0, 0, 0

        ### combine from all dataloaders
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
                ### get losses
                G_loss_b, G_loss_stim_b, G_loss_adv_b, D_loss_b, D_real_stim_loss_b, D_fake_stim_loss_b, stim_pred = get_training_losses(
                    model=model,
                    resp=dp["resp"],
                    stim=dp["stim"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    loss_fn=loss_fn,
                    config=config,
                    crop_win=config["crop_wins"][dp["data_key"]],
                )
                G_loss, G_loss_stim, G_loss_adv = G_loss + G_loss_b, G_loss_stim + G_loss_stim_b, G_loss_adv + G_loss_adv_b
                D_loss, D_real_stim_loss, D_fake_stim_loss = D_loss + D_loss_b, D_real_stim_loss + D_real_stim_loss_b, D_fake_stim_loss + D_fake_stim_loss_b
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
                G_loss = G_loss + model.get_additional_loss(data_key=dp["data_key"])
                n_dps += 1
                seen_data_keys.add(dp["data_key"])
            
        if n_dps == 0: break

        ### update
        # update generator
        G_loss /= n_dps
        G_loss, G_mean_abs_grad_first_layer, G_mean_abs_grad_last_layer = update_G(model=model, G_loss=G_loss, config=config, data_keys=seen_data_keys)

        # update discriminator
        D_loss /= n_dps
        D_loss, D_mean_abs_grad_first_layer, D_mean_abs_grad_last_layer = update_D(model=model, D_loss=D_loss, config=config, data_keys=seen_data_keys)

        ### log
        G_loss_stim, G_loss_adv, D_real_stim_loss, D_fake_stim_loss = G_loss_stim / n_dps, G_loss_adv / n_dps, D_real_stim_loss / n_dps, D_fake_stim_loss / n_dps
        history["G_mean_abs_grad_first_layer"].append(G_mean_abs_grad_first_layer)
        history["G_mean_abs_grad_last_layer"].append(G_mean_abs_grad_last_layer)
        history["D_mean_abs_grad_first_layer"].append(D_mean_abs_grad_first_layer)
        history["D_mean_abs_grad_last_layer"].append(D_mean_abs_grad_last_layer)
        history["D_loss"].append(D_loss.item())
        history["G_loss"].append(G_loss.item())
        history["G_loss_stim"].append(G_loss_stim.item())
        history["G_loss_adv"].append(G_loss_adv.item())
        history["D_loss_real"].append(D_real_stim_loss.item())
        history["D_loss_fake"].append(D_fake_stim_loss.item())
        if wdb_run is not None:
            wdb_run.log(
                {m: history[m][-1] for m in ("D_loss", "G_loss", "G_loss_stim", "G_loss_adv", "D_loss_real", "D_loss_fake")},
                commit=wdb_commit,
            )
        if batch_idx % log_freq == 0:
            print(
                f"  [{batch_idx * len(dp['resp'])}/{n_batches * len(dp['resp'])} "
                f"({100. * batch_idx / n_batches:.0f}%)] "
                f"G-loss: {G_loss.item():.3f} (stim: {G_loss_stim.item():.3f}, adv: {G_loss_adv.item():.3f})   "
                f"D-loss: {D_loss.item():.3f} (real: {D_real_stim_loss.item():.3f}, fake: {D_fake_stim_loss.item():.3f})"
            )
        batch_idx += 1

    return history

# @timeit
# def val(model, dataloaders, loss_fn, crop_wins=None):
#     model.eval()
#     val_loss = 0
#     n_samples = 0

#     ### is loss_fn FID?
#     is_fid = False
#     if type(loss_fn) == str and loss_fn.lower() == "fid":
#         is_fid = True
#         preds, targets = defaultdict(list), defaultdict(list)

#     with torch.no_grad():
#         for k, dl in dataloaders.items():
#             for batch_idx, b in enumerate(dl):
#                 ### combine from all data keys
#                 for dp in b:
#                     ### predict
#                     stim_pred = model(dp["resp"], data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"])

#                     ### calc loss/FID
#                     if is_fid:
#                         preds[dp["data_key"]].append(crop(stim_pred, crop_wins[dp["data_key"]]).cpu())
#                         targets[dp["data_key"]].append(crop(dp["stim"], crop_wins[dp["data_key"]]).cpu())
#                     else:
#                         val_loss += loss_fn(stim_pred, dp["stim"], phase="val", data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"]).item()
#                         n_samples += dp["resp"].shape[0]

#     ### finalize the val loss
#     if is_fid:
#         for data_key in preds.keys():
#             fid = FID(inp_standardized=False, device="cpu")
#             val_loss += fid(
#                 pred_imgs=torch.cat(preds[data_key], dim=0),
#                 gt_imgs=torch.cat(targets[data_key], dim=0)
#             )
#         val_loss /= len(preds.keys()) # average over data keys
#     else:
#         val_loss /= n_samples

#     return val_loss

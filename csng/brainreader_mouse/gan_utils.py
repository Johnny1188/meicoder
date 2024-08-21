import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict

import csng
from csng.utils import standardize, normalize, crop
from csng.losses import FID


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

    return G_loss, G_loss_stim, G_loss_adv, D_loss, real_stim_loss, fake_stim_loss


def update_G(model, G_loss, config, data_keys=None):
    model.core.G_optim.zero_grad()

    ### regularization
    if config["decoder"]["G_reg"]["l2"] > 0:
        G_loss += sum(p.pow(2).sum() for n,p in model.core.G.named_parameters() \
            if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l2"]
        G_loss += sum(p.pow(2).sum() for n,p in model.readins.named_parameters() \
            if p.requires_grad and "bias" not in n and (data_keys is None or sum(dk in n for dk in data_keys) == 1)) * config["decoder"]["G_reg"]["l2"]
    if config["decoder"]["G_reg"]["l1"] > 0:
        G_loss += sum(p.abs().sum() for n,p in model.core.G.named_parameters() \
            if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l1"]
        G_loss += sum(p.abs().sum() for n,p in model.readins.named_parameters() \
            if p.requires_grad and "bias" not in n and (data_keys is None or sum(dk in n for dk in data_keys) == 1)) * config["decoder"]["G_reg"]["l1"]
    G_loss.backward()

    ### clip gradients
    for n, p in model.core.G.named_parameters():
        if "head" in n and data_keys is not None and sum(dk in n for dk in data_keys) == 0:
            continue
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
            if p.requires_grad and "bias" not in n and ("head" not in n or sum(dk in n for dk in data_keys) == 1)) * config["decoder"]["D_reg"]["l2"]
    if config["decoder"]["D_reg"]["l1"] > 0:
        D_loss += sum(p.abs().sum() for n,p in model.core.D.named_parameters() \
            if p.requires_grad and "bias" not in n and ("head" not in n or sum(dk in n for dk in data_keys) == 1)) * config["decoder"]["D_reg"]["l1"]
    D_loss.backward()

    ### clip gradients
    for n, p in model.core.D.named_parameters():
        if "head" in n and data_keys is not None and sum(dk in n for dk in data_keys) == 0:
            continue
        p.grad.data.clamp_(-1., 1.)

    ### log
    D_mean_abs_grad_first_layer = torch.mean(torch.abs(model.core.D.layers[0].weight.grad)).item()
    if model.core.D.head is None:
        D_mean_abs_grad_last_layer = torch.mean(torch.abs(model.core.D.layers[-2].weight.grad)).item()
    else:
        D_mean_abs_grad_last_layer, div_by = 0, 0
        for n, p in model.core.D.head.named_parameters():
            if data_keys is not None and sum(dk in n for dk in data_keys) == 0:
                continue
            if "weight" in n:
                D_mean_abs_grad_last_layer += p.grad.abs().mean().item()
                div_by += 1
        D_mean_abs_grad_last_layer /= max(div_by, 1)

    ### step
    model.core.D_optim.step()

    return D_loss, D_mean_abs_grad_first_layer, D_mean_abs_grad_last_layer


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
                G_loss_b, G_loss_stim_b, G_loss_adv_b, D_loss_b, D_real_stim_loss_b, D_fake_stim_loss_b = get_training_losses(
                    model=model,
                    resp=dp["resp"],
                    stim=dp["stim"],
                    data_key=dp["data_key"],
                    neuron_coords=dp["neuron_coords"],
                    loss_fn=loss_fn,
                    config=config,
                    crop_win=config["data"][k]["crop_win"],
                )
                G_loss, G_loss_stim, G_loss_adv = G_loss + G_loss_b, G_loss_stim + G_loss_stim_b, G_loss_adv + G_loss_adv_b
                D_loss, D_real_stim_loss, D_fake_stim_loss = D_loss + D_loss_b, D_real_stim_loss + D_real_stim_loss_b, D_fake_stim_loss + D_fake_stim_loss_b
                # model.set_additional_loss(
                #     inp={
                #         "resp": dp["resp"],
                #         "stim": dp["stim"],
                #         "neuron_coords": dp["neuron_coords"],
                #         "pupil_center": dp["pupil_center"],
                #         "data_key": dp["data_key"],
                #     }, out={
                #         "stim_pred": stim_pred,
                #     },
                # )
                # loss += model.get_additional_loss(data_key=dp["data_key"])
                n_dps += 1
                seen_data_keys.add(dp["data_key"])

        if n_dps == 0: break

        ### update
        ##### update generator [START] #####
        G_loss /= n_dps
        G_loss, G_mean_abs_grad_first_layer, G_mean_abs_grad_last_layer = update_G(model=model, G_loss=G_loss, config=config, data_keys=seen_data_keys)
        ##### update generator [END] #####

        ##### update discriminator [START] #####
        D_loss /= n_dps
        D_loss, D_mean_abs_grad_first_layer, D_mean_abs_grad_last_layer = update_D(model=model, D_loss=D_loss, config=config, data_keys=seen_data_keys)
        ##### update discriminator [END] #####

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


def val(model, dataloaders, loss_fn, crop_wins=None):
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
                        preds[dp["data_key"]].append(crop(stim_pred, crop_wins[dp["data_key"]]).cpu())
                        targets[dp["data_key"]].append(crop(dp["stim"], crop_wins[dp["data_key"]]).cpu())
                    else:
                        val_loss += loss_fn(stim_pred, dp["stim"], phase="val", data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"]).item()
                        n_samples += dp["resp"].shape[0]
                        # loss = loss_fn(stim_pred, stim, data_key=data_key, phase="val").item()
                        # val_losses["total"] += loss
                        # val_losses[data_key] = loss if data_key not in val_losses else val_losses[data_key] + loss
                        # denom_data_keys[data_key] = denom_data_keys[data_key] + resp.shape[0] if data_key in denom_data_keys else resp.shape[0]
                        # n_samples += resp.shape[0]

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


# def get_all_data(config):
#     dls, neuron_coords = get_mouse_v1_data(config=config["data"])
#     if "syn_dataset_config" in config["data"] and config["data"]["syn_dataset_config"] is not None:
#         dls = append_syn_dataloaders(
#             dataloaders=dls,
#             config=config["data"]["syn_dataset_config"]
#         ) # append synthetic data
#     if "data_augmentation" in config["data"] and config["data"]["data_augmentation"] is not None:
#         dls = append_data_aug_dataloaders(
#             dataloaders=dls,
#             config=config["data"]["data_augmentation"],
#         )
#     return dls, neuron_coords

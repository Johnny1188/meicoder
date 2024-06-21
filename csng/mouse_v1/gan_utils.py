import os
import random
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict

from csng.utils import crop
from data_utils import (
    get_mouse_v1_data,
    append_syn_dataloaders,
    append_data_aug_dataloaders,
)
from csng.losses import FID



def train(model, dataloader, loss_fn, config, history, epoch, log_freq=100, wdb_run=None, wdb_commit=True):
    for k in ("G_mean_abs_grad_first_layer", "G_mean_abs_grad_last_layer", "D_mean_abs_grad_first_layer", "D_mean_abs_grad_last_layer",
        "D_loss", "G_loss", "G_loss_stim", "G_loss_adv", "D_loss_real", "D_loss_fake"):
        if k not in history.keys(): history[k] = []

    model.train()

    ### run
    for batch_idx, b in enumerate(dataloader):
        D_loss, G_loss = 0, 0

        ### combine losses from all data keys
        for data_key, stim, resp, neuron_coords, pupil_center in b:
            ### get the stimulus reconstruction from the generator
            stim_pred = model(
                resp,
                data_key=data_key,
                neuron_coords=neuron_coords,
                pupil_center=pupil_center
            )

            ### get discriminator loss
            # real stim (add noise to labels (uniform distribution between 0.xx and 1))
            real_stim_pred = model.core.D(crop(stim, config["crop_win"]))
            noisy_real_stim_labels = (
                1. - config["decoder"]["D_real_stim_labels_noise"] 
                + torch.rand_like(real_stim_pred) * config["decoder"]["D_real_stim_labels_noise"]
            )
            real_stim_loss = torch.mean((real_stim_pred - noisy_real_stim_labels)**2) * config["decoder"]["D_real_loss_mul"]
            # fake stim (add noise to labels (uniform distribution between 0 and noise level))
            fake_stim_pred = model.core.D(crop(stim_pred.detach(), config["crop_win"]))
            noisy_fake_stim_labels = torch.rand_like(fake_stim_pred) * config["decoder"]["D_fake_stim_labels_noise"]
            fake_stim_loss = torch.mean((fake_stim_pred - noisy_fake_stim_labels)**2) * config["decoder"]["D_fake_loss_mul"]
            D_loss = D_loss + real_stim_loss + fake_stim_loss

            ### get generator loss
            # fooling the discriminator
            fake_stim_pred_for_G = model.core.D(crop(stim_pred, config["crop_win"]))
            G_loss_adv = torch.mean((fake_stim_pred_for_G - 1.)**2) * config["decoder"]["G_adv_loss_mul"]
            # reconstruction quality
            G_loss_stim = config["decoder"]["G_stim_loss_mul"] * loss_fn(stim_pred, stim, data_key=data_key, phase="train", neuron_coords=neuron_coords, pupil_center=pupil_center)
            # additional loss (regularization etc.)
            model.set_additional_loss(
                inp={
                    "resp": resp,
                    "stim": stim,
                    "neuron_coords": neuron_coords,
                    "pupil_center": pupil_center,
                    "data_key": data_key,
                }, out={
                    "stim_pred": stim_pred,
                },
            )
            G_loss_additional = model.get_additional_loss(data_key=data_key)
            G_loss += G_loss_adv + G_loss_stim + G_loss_additional

        ##### update generator [START] #####
        model.core.G_optim.zero_grad()
        G_loss /= len(b)

        # regularization
        if config["decoder"]["G_reg"]["l2"] > 0:
            G_loss += sum(p.pow(2).sum() for n,p in model.core.G.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l2"]
            G_loss += sum(p.pow(2).sum() for n,p in model.readins.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l2"]
        if config["decoder"]["G_reg"]["l1"] > 0:
            G_loss += sum(p.abs().sum() for n,p in model.core.G.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l1"]
            G_loss += sum(p.abs().sum() for n,p in model.readins.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["G_reg"]["l1"]
        G_loss.backward()

        # clip gradients
        for p in model.core.G.parameters():
            p.grad.data.clamp_(-1., 1.)
        
        # log
        history["G_mean_abs_grad_first_layer"].append(torch.mean(torch.abs(model.core.G.layers[0].weight.grad)).item())
        history["G_mean_abs_grad_last_layer"].append(torch.mean(torch.abs(model.core.G.layers[-2].weight.grad)).item())

        # step
        model.core.G_optim.step()
        ##### update generator [END] #####


        ##### update discriminator [START] #####
        model.core.D_optim.zero_grad()
        D_loss /= len(b)

        # regularization
        if config["decoder"]["D_reg"]["l2"] > 0:
            D_loss += sum(p.pow(2).sum() for n,p in model.core.D.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["D_reg"]["l2"]
        if config["decoder"]["D_reg"]["l1"] > 0:
            D_loss += sum(p.abs().sum() for n,p in model.core.D.named_parameters() \
                if p.requires_grad and "bias" not in n) * config["decoder"]["D_reg"]["l1"]
        D_loss.backward()

        # clip gradients
        for p in model.core.D.parameters():
            p.grad.data.clamp_(-1., 1.)

        # log
        history["D_mean_abs_grad_first_layer"].append(torch.mean(torch.abs(model.core.D.layers[0].weight.grad)).item())
        history["D_mean_abs_grad_last_layer"].append(torch.mean(torch.abs(model.core.D.layers[-2].weight.grad)).item())

        # step
        model.core.D_optim.step()
        ##### update discriminator [END] #####


        ### log
        history["D_loss"].append(D_loss.item())
        history["G_loss"].append(G_loss.item())
        history["G_loss_stim"].append(G_loss_stim.item())
        history["G_loss_adv"].append(G_loss_adv.item())
        history["D_loss_real"].append(real_stim_loss.item())
        history["D_loss_fake"].append(fake_stim_loss.item())
        if wdb_run is not None:
            wdb_run.log(
                {m: history[m][-1] for m in ("D_loss", "G_loss", "G_loss_stim", "G_loss_adv", "D_loss_real", "D_loss_fake")},
                commit=wdb_commit,
            )
        if batch_idx % log_freq == 0:
            print(
                f"  [{batch_idx * len(resp)}/{len(dataloader) * len(resp)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)] "
                f"G-loss: {G_loss.item():.3f} (stim: {G_loss_stim.item():.3f}, adv: {G_loss_adv.item():.3f})   "
                f"D-loss: {D_loss.item():.3f} (real: {real_stim_loss.item():.3f}, fake: {fake_stim_loss.item():.3f})"
            )

    return history


def val(model, dataloader, loss_fn, only_data_keys=None, crop_win=None):
    model.eval()
    val_losses = {"total": 0}
    n_samples = 0
    denom_data_keys = {}

    ### is loss_fn FID?
    is_fid = False
    if type(loss_fn) == str and loss_fn.lower() == "fid":
        is_fid = True
        preds, targets = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for b in dataloader:
            ### combine from all data keys
            for data_key, stim, resp, neuron_coords, pupil_center in b:
                if only_data_keys is not None and data_key not in only_data_keys:
                    continue
                stim_pred = model(
                    resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                )

                ### calc loss/FID
                if is_fid:
                    preds[data_key].append(crop(stim_pred, crop_win).cpu())
                    targets[data_key].append(crop(stim, crop_win).cpu())
                else:
                    loss = loss_fn(stim_pred, stim, data_key=data_key, phase="val").item()
                    val_losses["total"] += loss
                    val_losses[data_key] = loss if data_key not in val_losses else val_losses[data_key] + loss
                    denom_data_keys[data_key] = denom_data_keys[data_key] + resp.shape[0] if data_key in denom_data_keys else resp.shape[0]
                    n_samples += resp.shape[0]

    ### finalize the val loss
    if is_fid:
        for data_key in preds.keys():
            fid = FID(inp_standardized=False, device="cpu")
            val_losses[data_key] = fid(
                pred_imgs=torch.cat(preds[data_key], dim=0),
                gt_imgs=torch.cat(targets[data_key], dim=0)
            )
            val_losses["total"] += val_losses[data_key]
        val_losses["total"] /= len(preds.keys()) # average over data keys
    else:
        val_losses["total"] /= n_samples
        for k in denom_data_keys:
            val_losses[k] /= denom_data_keys[k]

    return val_losses


def get_all_data(config):
    dls, neuron_coords = get_mouse_v1_data(config=config["data"])
    if "syn_dataset_config" in config["data"] and config["data"]["syn_dataset_config"] is not None:
        dls = append_syn_dataloaders(
            dataloaders=dls,
            config=config["data"]["syn_dataset_config"]
        ) # append synthetic data
    if "data_augmentation" in config["data"] and config["data"]["data_augmentation"] is not None:
        dls = append_data_aug_dataloaders(
            dataloaders=dls,
            config=config["data"]["data_augmentation"],
        )
    return dls, neuron_coords

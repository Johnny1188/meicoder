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

from csng.mouse_v1.data_utils import (
    get_mouse_v1_data,
    append_syn_dataloaders,
    append_data_aug_dataloaders,
)


def train(model, dataloader, opter, loss_fn, verbose=True):
    model.train()
    train_loss = 0
    n_batches = len(dataloader)

    ### run
    for batch_idx, b in enumerate(dataloader):
        opter.zero_grad()
        loss = 0

        ### combine from all data keys
        for data_key, stim, resp, neuron_coords, pupil_center in b:
            ### get loss
            stim_pred = model(
                resp,
                data_key=data_key,
                neuron_coords=neuron_coords,
                pupil_center=pupil_center
            )
            loss += loss_fn(stim_pred, stim, resp=resp, data_key=data_key, phase="train", neuron_coords=neuron_coords, pupil_center=pupil_center)
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
            loss += model.get_additional_loss(data_key=data_key)

        ### update
        loss /= len(b)
        loss.backward()
        opter.step()

        ### log
        train_loss += loss.item()
        if verbose and batch_idx % 100 == 0:
            print(f"Training progress: [{batch_idx}/{n_batches} ({100. * batch_idx / n_batches:.0f}%)]"
                  f"  Loss: {loss.item():.6f}")

    train_loss /= n_batches
    return train_loss


def val(model, dataloader, loss_fn, only_data_keys=None):
    model.eval()
    val_losses = {"total": 0}
    n_samples = 0
    denom_data_keys = {}
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
                loss = loss_fn(stim_pred, stim, data_key=data_key, phase="val").item()
                val_losses["total"] += loss
                val_losses[data_key] = loss if data_key not in val_losses else val_losses[data_key] + loss
                denom_data_keys[data_key] = denom_data_keys[data_key] + resp.shape[0] if data_key in denom_data_keys else resp.shape[0]
                n_samples += resp.shape[0]

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

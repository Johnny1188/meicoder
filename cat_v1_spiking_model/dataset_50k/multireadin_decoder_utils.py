import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import plot_losses, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, crop
from csng.losses import SSIMLoss, MSELossWithCrop, Loss
from csng.readins import MultiReadIn, FCReadIn, ConvReadIn

from cat_v1_spiking_model.dataset_50k.data import (
    prepare_v1_dataloaders,
    SyntheticDataset,
    BatchPatchesDataLoader,
    MixedBatchLoader,
    PerSampleStoredDataset,
)

lt.monkey_patch()

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
print(f"{DATA_PATH=}")

def train(model, dataloader, opter, loss_fn, config, verbose=True):
    model.train()
    train_loss = 0
    n_batches = len(dataloader)
    
    ### run
    # for batch_idx, (stim, resp, coords, ori) in enumerate(dataloader):
    for batch_idx, b in enumerate(dataloader):
        opter.zero_grad()
        loss = 0

        ### combine from all data keys
        for data_key, (stim, resp, neuron_coords) in b.items():
            ### data
            stim = stim.to(config["device"])
            resp = resp.to(config["device"])
            neuron_coords = neuron_coords.float().to(config["device"])
        
            ### train
            stim_pred = model(resp, data_key=data_key, neuron_coords=neuron_coords)
            loss += loss_fn(stim_pred, stim, data_key=data_key, neuron_coords=neuron_coords, phase="train")
            model.set_additional_loss(
                inp={
                    "resp": resp,
                    "stim": stim,
                    "neuron_coords": neuron_coords,
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


def val(model, dataloader, loss_fn, config):
    model.eval()
    val_loss = 0
    n_samples = 0

    with torch.no_grad():
        for batch_idx, b in enumerate(dataloader):
            ### combine from all data keys
            for data_key, (stim, resp, neuron_coords) in b.items():
                ### data
                stim = stim.to(config["device"])
                resp = resp.to(config["device"])
                neuron_coords = neuron_coords.float().to(config["device"])

                ### predict
                stim_pred = model(resp, data_key=data_key, neuron_coords=neuron_coords)
                val_loss += loss_fn(stim_pred, stim, data_key=data_key, neuron_coords=neuron_coords, phase="val").item()
                n_samples += resp.shape[0]

    val_loss /= n_samples
    return val_loss


def get_dataloaders(config, dataloaders, only_cat_v1_eval=True):
    ### get dataloaders to mix
    dataloaders_to_mix = [dl for dl in dataloaders.values()]

    ### get dataloaders
    train_dataloader = MixedBatchLoader(
        dataloaders=[dl["train"] for dl in dataloaders_to_mix],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1" for _ in dataloaders_to_mix],
        return_pupil_center=False,
    )
    val_dataloader = MixedBatchLoader(
        dataloaders=[dl["val"] for dl in dataloaders_to_mix] \
            if not only_cat_v1_eval else [dataloaders["cat_v1"]["val"]],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1" for _ in dataloaders_to_mix],
        return_pupil_center=False,
    )
    test_dataloader = MixedBatchLoader(
        dataloaders=[dl["test"] for dl in dataloaders_to_mix] \
            if not only_cat_v1_eval else [dataloaders["cat_v1"]["test"]],
        mixing_strategy=config["data"]["mixing_strategy"],
        device=config["device"],
        data_keys=["cat_v1" for _ in dataloaders_to_mix],
        return_pupil_center=False,
    )

    return train_dataloader, val_dataloader, test_dataloader

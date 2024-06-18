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
from nnfabrik.builder import get_data

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import plot_losses, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, crop
from csng.losses import SSIMLoss, Loss
from csng.readins import MultiReadIn, FCReadIn, ConvReadIn

from csng.mouse_v1.data_utils import get_mouse_v1_data, append_syn_dataloaders, append_data_aug_dataloaders, average_test_multitrial
from csng.cat_v1.data import prepare_v1_dataloaders


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


def val(model, dataloaders, loss_fn, config):
    model.eval()
    val_loss = 0
    n_samples = 0

    with torch.no_grad():
        for k, dl in dataloaders.items():
            for batch_idx, b in enumerate(dl):
                ### combine from all data keys
                for dp in b:
                    ### predict
                    stim_pred = model(dp["resp"], data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"])
                    val_loss += loss_fn(stim_pred, dp["stim"], phase="val", data_key=dp["data_key"], neuron_coords=dp["neuron_coords"], pupil_center=dp["pupil_center"]).item()
                    n_samples += dp["resp"].shape[0]

    val_loss /= n_samples
    return val_loss


def get_dataloaders(config):
    dls = dict(train=dict(), val=dict(), test=dict())
    neuron_coords = dict()

    ### mouse v1 - base
    if "mouse_v1" in config["data"] and config["data"]["mouse_v1"] is not None:
        ### get dataloaders
        _dataloaders = get_data(config["data"]["mouse_v1"]["dataset_fn"], config["data"]["mouse_v1"]["dataset_config"])

        if config["data"]["mouse_v1"]["average_test_multitrial"]:
            _dataloaders["test"] = average_test_multitrial(_dataloaders["test"], config["data"])

        m_dls = {
            "mouse_v1": {
                "train": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["train"][data_key] for data_key in _dataloaders["train"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    max_batches=config["data"].get("max_training_batches"),
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_train"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "val": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["validation"][data_key] for data_key in _dataloaders["validation"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_val"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "test": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["test"][data_key] for data_key in _dataloaders["test"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_test"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "test_no_resp": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["final_test"][data_key] for data_key in _dataloaders["final_test"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                )
            }
        }
        
        ### get cell coordinates
        _neuron_coords = {
            data_key: torch.tensor(d.neurons.cell_motor_coordinates, dtype=torch.float32, device=config["data"]["mouse_v1"]["device"])
            for data_key, d in zip(list(_dataloaders["train"].keys()), [_dl.dataset for _dl in _dataloaders["train"].values()])
        }
        if config["data"]["mouse_v1"]["normalize_neuron_coords"]: # normalize coordinates to [-1, 1]
            for data_key in _neuron_coords.keys():
                ### normalize x,y,z separately
                for dim_idx in range(_neuron_coords[data_key].shape[-1]):
                    _neuron_coords[data_key][:, dim_idx] = \
                        (_neuron_coords[data_key][:, dim_idx] - _neuron_coords[data_key][:, dim_idx].min()) \
                        / (_neuron_coords[data_key][:, dim_idx].max() - _neuron_coords[data_key][:, dim_idx].min()) * 2 - 1

        ### assign neuron_coords to dataloaders
        for dl_type in ["train", "val", "test", "test_no_resp"]:
            m_dls["mouse_v1"][dl_type].neuron_coords = _neuron_coords
        neuron_coords["mouse_v1"] = _neuron_coords

        ### mouse v1 - synthetic data
        if "syn_dataset_config" in config["data"] and config["data"]["syn_dataset_config"] is not None:
            raise NotImplementedError
            m_dls = append_syn_dataloaders(
                dataloaders=m_dls,
                config=config["data"]["syn_dataset_config"]
            )

        ### mouse v1 - data augmentation
        if "data_augmentation" in config["data"] and config["data"]["data_augmentation"] is not None:
            raise NotImplementedError
            m_dls = append_data_aug_dataloaders(
                dataloaders=m_dls,
                config=config["data"]["data_augmentation"],
            )

        ### add to dls
        for tier in ("train", "val", "test"):
            dls[tier]["mouse_v1"] = m_dls["mouse_v1"][tier]

    ### cat v1
    if "cat_v1" in config["data"]:
        c_dls = prepare_v1_dataloaders(**config["data"]["cat_v1"]["dataset_config"])

        ### get neuron coordinates
        torch.allclose(c_dls["train"].dataset[0].neuron_coords, c_dls["train"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["train"].dataset[-1].neuron_coords, c_dls["val"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[0].neuron_coords, c_dls["val"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[-1].neuron_coords, c_dls["test"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["test"].dataset[0].neuron_coords, c_dls["test"].dataset[-1].neuron_coords), \
            "Neuron coordinates must be the same for all samples in the dataset"
        neuron_coords["cat_v1"] = c_dls["train"].dataset[0].neuron_coords.float().to(config["device"])

        ### add to dls
        for tier in ("train", "val", "test"):
            dls[tier]["cat_v1"] = MixedBatchLoaderV2(
                dataloaders=[c_dls[tier]],
                neuron_coords={"cat_v1": neuron_coords["cat_v1"]},
                mixing_strategy=config["data"]["mixing_strategy"],
                max_batches=config["data"].get("max_training_batches"),
                data_keys=["cat_v1"],
                return_data_key=True,
                return_pupil_center=False,
                device=config["device"],
            )

    return dls, neuron_coords


class MixedBatchLoaderV2:
    def __init__(
        self,
        dataloaders,
        neuron_coords=None,
        mixing_strategy="sequential",
        max_batches=None,
        data_keys=None,
        return_data_key=True,
        return_pupil_center=True,
        return_neuron_coords=True,
        device="cpu",
    ):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel_min', 'parallel_max'], but got {mixing_strategy}"

        self.dataloaders = dataloaders
        self.neuron_coords = neuron_coords
        self.data_keys = data_keys
        self.dataloader_iters = {dl_idx: {"dl": iter(dataloader)} for dl_idx, dataloader in enumerate(dataloaders)}
        if data_keys is not None:
            assert len(data_keys) == len(dataloaders), f"len(data_keys) must be equal to len(dataloaders), but got {len(data_keys)} and {len(dataloaders)}"
            for dl_idx, data_key in zip(self.dataloader_iters.keys(), data_keys):
                self.dataloader_iters[dl_idx]["data_key"] = data_key
        self.dataloaders_left = list(self.dataloader_iters.keys())
        self.n_dataloaders = len(self.dataloader_iters)
        self.mixing_strategy = mixing_strategy
        self.max_batches = max_batches
        
        self.return_data_key = return_data_key
        self.return_pupil_center = return_pupil_center
        self.return_neuron_coords = return_neuron_coords
        
        self.device = device
        self.batch_idx = 0
        
        if self.mixing_strategy == "sequential":
            self.n_batches = sum([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0
        if self.max_batches is not None:
            self.n_batches = min(self.n_batches, self.max_batches)

        self.datasets = []
        for dl in dataloaders:
            if hasattr(dl, "dataset"):
                self.datasets.append(dl.dataset)
            else:
                self.datasets.append(dl)

    def add_dataloader(self, dataloader, neuron_coords=None, data_key=None):
        self.dataloaders.append(dataloader)
        dl_idx = len(self.dataloaders)
        self.dataloader_iters[dl_idx] = {"dl": iter(dataloader)}
        if data_key is not None:
            self.dataloader_iters[dl_idx]["data_key"] = data_key
            if type(self.data_keys) == list:
                self.data_keys.append(data_key)
        self.dataloaders_left.append(dl_idx)
        self.n_dataloaders += 1
        if self.mixing_strategy == "sequential":
            self.n_batches += len(dataloader)
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max(self.n_batches, len(dataloader))
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min(self.n_batches, len(dataloader)) \
                if self.n_batches > 0 else len(dataloader)
        if hasattr(dataloader, "dataset"):
            self.datasets.append(dataloader.dataset)
        else:
            self.datasets.append(dataloader)

        if neuron_coords is not None:
            for _data_key, coords in neuron_coords.items():
                if _data_key in self.neuron_coords.keys() \
                    and not torch.equal(self.neuron_coords[_data_key], coords.to(self.device)):
                    print(f"[WARNING]: neuron_coords for data_key {_data_key} already exists and are not the same. Overwriting.")
                self.neuron_coords[_data_key] = coords.to(self.device)

    def _get_sequential(self):
        to_return = []
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                datapoint = next(self.dataloader_iters[dl_idx]["dl"])
                to_return.append(dict(data_key=None, stim=None, resp=None, neuron_coords=None, pupil_center=None))
                to_return[-1]["stim"] = datapoint.images.to(self.device)
                to_return[-1]["resp"] = datapoint.responses.to(self.device)
                if self.return_data_key:
                    to_return[-1]["data_key"] = self.dataloader_iters[dl_idx]["data_key"]
                if self.return_neuron_coords:
                    _neuron_coords = self.neuron_coords[self.dataloader_iters[dl_idx]["data_key"]]
                    to_return[-1]["neuron_coords"] = _neuron_coords.to(self.device)
                if self.return_pupil_center:
                    to_return[-1]["pupil_center"] = datapoint.pupil_center.to(self.device)
                break
            except StopIteration:
                ### no more data in this dataloader
                self.dataloaders_left = [_dl_idx for _dl_idx in self.dataloaders_left if _dl_idx != dl_idx]
                del self.dataloader_iters[dl_idx]
                self.n_dataloaders -= 1
                if self.n_dataloaders == 0:  # no more data
                    break
                else:
                    continue

        return to_return
    
    def _get_parallel(self):
        empty_dataloader_idxs = set()
        to_return = []
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                datapoint = next(dataloader_iter["dl"])
                to_return.append(dict(data_key=None, stim=None, resp=None, neuron_coords=None, pupil_center=None))
                to_return[-1]["stim"] = datapoint.images.to(self.device)
                to_return[-1]["resp"] = datapoint.responses.to(self.device)
                if self.return_data_key:
                    to_return[-1]["data_key"] = dataloader_iter["data_key"]
                if self.return_neuron_coords:
                    to_return[-1]["neuron_coords"] = self.neuron_coords[dataloader_iter["data_key"]].to(self.device)
                if self.return_pupil_center and "pupil_center" in datapoint._fields:
                    to_return[-1]["pupil_center"] = datapoint.pupil_center.to(self.device)
            except StopIteration:
                ### no more data in this dataloader
                if self.mixing_strategy == "parallel_min":
                    ### end the whole loop
                    empty_dataloader_idxs = set(self.dataloader_iters.keys())
                elif self.mixing_strategy == "parallel_max":
                    ### continue with the remaining ones
                    empty_dataloader_idxs.add(dl_idx)
                else:
                    raise NotImplementedError

        ### remove empty dataloaders
        if len(empty_dataloader_idxs) > 0:
            for dl_idx_to_remove in empty_dataloader_idxs:
                del self.dataloader_iters[dl_idx_to_remove]
            self.n_dataloaders = len(self.dataloader_iters)

        return to_return

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_idx += 1
        if self.mixing_strategy == "sequential":
            out = self._get_sequential()
        elif self.mixing_strategy in ("parallel_min", "parallel_max"):
            out = self._get_parallel()
        else:
            raise NotImplementedError

        if len(out) == 0 or (self.max_batches is not None and self.batch_idx > self.max_batches):
            raise StopIteration

        return out

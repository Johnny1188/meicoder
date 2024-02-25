import os
from collections import OrderedDict, namedtuple
import random
import pickle

import numpy as np
import pandas as pd
import skimage.transform
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision


import torch

class MixedBatchLoader:
    """
    A dataloader that interleaves or mixes batches from multiple dataloaders.

    Args:
        dataloaders (list): A list of dataloaders to interleave or mix batches from.
        mixing_strategy (str): The strategy to use for mixing batches. Must be one of ["sequential", "parallel_min", "parallel_max"].
        device (str): The device to load the batches onto. Default is "cpu".

    Raises:
        AssertionError: If mixing_strategy is not one of ["sequential", "parallel_min", "parallel_max"].

    Attributes:
        dataloader_iters (list): A list of iterators for each dataloader.
        n_dataloaders (int): The number of dataloaders.
        mixing_strategy (str): The strategy used for mixing batches.
        device (str): The device to load the batches onto.
        batch_idx (int): The current batch index.
        n_batches (int): The total number of batches.

    Methods:
        _get_sequential(): Interleaves batches from multiple dataloaders.
        _get_parallel(): Mixes batches from multiple dataloaders.
        __len__(): Returns the total number of batches.
        __iter__(): Returns the iterator object.
        __next__(): Returns the next batch of data.

    """
    
    def __init__(self, dataloaders, mixing_strategy="sequential", data_keys=None, return_data_key=False, device="cpu"):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel'], but got {mixing_strategy}"

        self.dataloaders = dataloaders
        self.data_keys = data_keys
        self.dataloader_iters = {dl_idx: {"dl": iter(dataloader)} for dl_idx, dataloader in enumerate(dataloaders)}
        if data_keys is not None:
            assert len(data_keys) == len(dataloaders), f"len(data_keys) must be equal to len(dataloaders), but got {len(data_keys)} and {len(dataloaders)}"
            for dl_idx, data_key in zip(self.dataloader_iters.keys(), data_keys):
                self.dataloader_iters[dl_idx]["data_key"] = data_key
        self.dataloaders_left = list(self.dataloader_iters.keys())
        self.n_dataloaders = len(self.dataloader_iters)
        self.mixing_strategy = mixing_strategy
        self.return_data_key = return_data_key
        self.device = device
        self.batch_idx = 0
        
        if self.mixing_strategy == "sequential":
            self.n_batches = sum([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min([len(dataloader) for dataloader in dataloaders])

        self.datasets = []
        for dl in dataloaders:
            if hasattr(dl, "dataset"):
                self.datasets.append(dl.dataset)
            else:
                self.datasets.append(dl)


    def add_dataloader(self, dataloader, data_key=None):
        """
        Adds a dataloader to the list of dataloaders.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to add.
            data_key (str): The key to use for the dataloader. If None, the dataloader will be added with a numeric key.
        """
        self.dataloaders.append(dataloader)
        dl_idx = len(self.dataloaders)
        self.dataloader_iters[dl_idx] = {"dl": iter(dataloader)}
        if data_key is not None:
            self.dataloader_iters[dl_idx]["data_key"] = data_key
            if type(self.data_keys) == list:
                self.data_keys.append(data_key)
        self.dataloaders_left.append(dl_idx)
        self.n_dataloaders += 1
        self.n_batches += len(dataloader)
        if hasattr(dataloader, "dataset"):
            self.datasets.append(dataloader.dataset)
        else:
            self.datasets.append(dataloader)


    def _get_sequential(self):
        """
        Interleaves batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                stim, resp = next(self.dataloader_iters[dl_idx]["dl"])
                break
            except StopIteration:
                self.dataloaders_left = [_dl_idx for _dl_idx in self.dataloaders_left if _dl_idx != dl_idx]
                del self.dataloader_iters[dl_idx]
                self.n_dataloaders -= 1
                if self.n_dataloaders == 0:
                    if self.return_data_key:
                        return None, None, None
                    return None, None
                else:
                    continue

        if self.return_data_key:
            return stim.to(self.device), resp.to(self.device), self.dataloader_iters[dl_idx]["data_key"]
        return stim.to(self.device), resp.to(self.device)
    
    def _get_parallel(self):
        """
        Mixes batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        stim, resp = [], []
        empty_dataloader_idxs = set()
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                _stim, _resp = next(dataloader_iter["dl"])
                stim.append(_stim.to(self.device))
                resp.append(_resp.to(self.device))
            except StopIteration:
                if self.mixing_strategy == "parallel_min":
                    ### if a single dataloader ends, end the whole loop
                    # _ = [empty_dataloader_idxs.add(_dl_idx) for _dl_idx in range(0, self.n_dataloaders)]
                    empty_dataloader_idxs = set(self.dataloader_iters.keys())
                elif self.mixing_strategy == "parallel_max":
                    ### if a single dataloader ends, continue with the remaining ones
                    # empty_dataloader_idxs.add(dl_idx)
                    empty_dataloader_idxs.add(dl_idx)
                else:
                    raise NotImplementedError
        
        ### remove empty dataloaders
        if len(empty_dataloader_idxs) > 0:
            # new_dataloader_iters = []
            # for dl_idx, dataloader_iter in enumerate(self.dataloader_iters):
            #     if dl_idx not in empty_dataloader_idxs:
            #         new_dataloader_iters.append(dataloader_iter)
            # self.dataloader_iters = new_dataloader_iters
            for dl_idx_to_remove in empty_dataloader_idxs:
                del self.dataloader_iters[dl_idx_to_remove]
            self.n_dataloaders = len(self.dataloader_iters)

        if len(stim) == 0:
            if self.return_data_key:
                return None, None, None
            return None, None

        ### concatenate
        stim = torch.cat(stim, dim=0)
        resp = torch.cat(resp, dim=0)
        if self.return_data_key:
            return stim, resp, None
        return stim, resp

    def __len__(self):
        """
        Returns the total number of batches.

        Returns:
            int: The total number of batches.
        """
        return self.n_batches

    def __iter__(self):
        """
        Returns the iterator object.

        Returns:
            MixedBatchLoader: The iterator object.
        """
        return self

    def __next__(self):
        """
        Returns the next batch of data.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        self.batch_idx += 1
        if self.mixing_strategy == "sequential":
            out = self._get_sequential()
        elif self.mixing_strategy in ("parallel_min", "parallel_max"):
            out = self._get_parallel()
        else:
            raise NotImplementedError
        
        if out[0] is None: # no more data
            raise StopIteration
        return out
import os
import pickle
import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from collections import namedtuple

from csng.utils.data import NumpyToTensor, NumpyImageCrop, NumpyImageResize


def get_cat_v1_dataloaders(
        train_path,
        val_path,
        test_path,
        image_size,
        crop,
        batch_size=64,
        return_coords=False,
        return_ori=False,
        cached=False,
        coords_ori_filepath=None,
        stim_normalize_mean=None,
        stim_normalize_std=None,
        resp_normalize_mean=None,
        resp_normalize_std=None,
        clamp_neg_resp=False,
        training_sample_idxs=None,
        stim_keys=("stim",),
        resp_keys=("resp",),
        verbose=False,
        device="cpu",
    ):
    ### prepare stimulus transforms
    if image_size != -1 and crop:
        stim_transform = [
            NumpyImageCrop(image_size),
            NumpyToTensor(device=device)
        ]
    elif image_size != -1 and not crop:
        stim_transform = [
            NumpyImageResize(image_size),
            NumpyToTensor(device=device)
        ]
    else:
        stim_transform = [
            NumpyToTensor(device=device),
            lambda x: np.expand_dims(x, 0)
        ]
    if stim_normalize_mean is not None and stim_normalize_std is not None:
        stim_transform.append(torchvision.transforms.Normalize(mean=stim_normalize_mean, std=stim_normalize_std))
    stim_transform = torchvision.transforms.Compose(stim_transform)

    ### prepare response transforms
    resp_transform = [NumpyToTensor(device=device)]
    if resp_normalize_mean is not None and resp_normalize_std is not None:
        if clamp_neg_resp:
            print("[WARNING]: clamp_neg_resp is True, but response normalization is also applied. This may lead to negative responses after normalization which will be clamped to 0.")
        resp_transform.append(torchvision.transforms.Lambda(lambda x: (x - resp_normalize_mean) / resp_normalize_std))
    elif resp_normalize_mean is not None:
        if clamp_neg_resp:
            print("[WARNING]: clamp_neg_resp is True, but response normalization is also applied. This may lead to negative responses after normalization which will be clamped to 0.")
        resp_transform.append(torchvision.transforms.Lambda(lambda x: x - resp_normalize_mean))
    elif resp_normalize_std is not None:
        resp_transform.append(torchvision.transforms.Lambda(lambda x: x / resp_normalize_std))
    resp_transform = torchvision.transforms.Compose(resp_transform)

    ### prepare datasets
    train_dataset = PerSampleStoredDataset(
        dataset_dir=train_path,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
        stim_keys=stim_keys,
        resp_keys=resp_keys,
        return_coords=return_coords,
        return_ori=return_ori,
        coords_ori_filepath=coords_ori_filepath,
        clamp_neg_resp=clamp_neg_resp,
        sample_idxs=training_sample_idxs,
    )
    val_dataset = PerSampleStoredDataset(
        dataset_dir=val_path,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
        stim_keys=stim_keys,
        resp_keys=resp_keys,
        return_coords=return_coords,
        return_ori=return_ori,
        coords_ori_filepath=coords_ori_filepath,
        clamp_neg_resp=clamp_neg_resp,
    )
    test_dataset = PerSampleStoredDataset(
        dataset_dir=test_path,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
        stim_keys=stim_keys,
        resp_keys=resp_keys,
        return_coords=return_coords,
        return_ori=return_ori,
        coords_ori_filepath=coords_ori_filepath,
        average_over_repeats=True,
        clamp_neg_resp=clamp_neg_resp,
    )
    if cached:
        train_dataset = CachedDataset(train_dataset)
        val_dataset = CachedDataset(val_dataset)
        test_dataset = CachedDataset(test_dataset)

    if verbose:
        print(f"Train dataset size: {len(train_dataset)}. Validation dataset size: {len(val_dataset)}. Test dataset size: {len(test_dataset)}.")

    ### create data loaders
    dls = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=4,
            # pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
            drop_last=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True,
            drop_last=True,
        )
    }

    return dls


class PerSampleStoredDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        stim_transform=None,
        resp_transform=None,
        stim_keys=("stim",),
        resp_keys=("resp",),
        return_coords=False,
        return_ori=False,
        coords_ori_filepath=None,
        average_over_repeats=False,
        clamp_neg_resp=False,
        sample_idxs=None,
        dataset_shuffle_seed=None,
    ):
        self.dataset_dir = dataset_dir
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
        self.file_names = np.array([
            f_name for f_name in os.listdir(self.dataset_dir)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ])
        if dataset_shuffle_seed is None:
            self.file_names = np.sort(self.file_names)
        else:
            np.random.default_rng(dataset_shuffle_seed).shuffle(self.file_names)
        self.sample_idxs = sample_idxs
        if sample_idxs is not None:
            self.file_names = self.file_names[self.sample_idxs]                
        self.stim_keys = stim_keys
        self.resp_keys = resp_keys
        self.average_over_repeats = average_over_repeats
        self.clamp_neg_resp = clamp_neg_resp

        self.return_coords = return_coords
        self.return_ori = return_ori
        self.coords, self.ori = None, None
        if return_coords or return_ori:
            assert coords_ori_filepath is not None, "coords_ori_filepath must be provided if return_coords or return_ori is True"
            with open(coords_ori_filepath, "rb") as f:
                pos_ori_file = pickle.load(f)
                self.coords = {
                    "V1_Exc_L23": np.concatenate((pos_ori_file["V1_Exc_L23"]["pos_x"].reshape(-1,1), pos_ori_file["V1_Exc_L23"]["pos_y"].reshape(-1,1)), axis=1),
                    "V1_Inh_L23": np.concatenate((pos_ori_file["V1_Inh_L23"]["pos_x"].reshape(-1,1), pos_ori_file["V1_Inh_L23"]["pos_y"].reshape(-1,1)), axis=1),
                }
                self.coords["all"] = np.concatenate((self.coords["V1_Exc_L23"], self.coords["V1_Inh_L23"]), axis=0)
                for k in self.coords:
                    self.coords[k] = torch.from_numpy(self.coords[k]).float()
                self.ori = {
                    "V1_Exc_L23": np.array(pos_ori_file["V1_Exc_L23"]["ori"]),
                    "V1_Inh_L23": np.array(pos_ori_file["V1_Inh_L23"]["ori"]),
                }
                self.ori["all"] = np.concatenate((self.ori["V1_Exc_L23"], self.ori["V1_Inh_L23"]), axis=0)
                for k in self.ori:
                    self.ori[k] = torch.from_numpy(self.ori[k]).float()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        with open(os.path.join(self.dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)
            stimuli = np.concatenate([data[key] for key in self.stim_keys], axis=0)
            if self.average_over_repeats:
                responses = np.concatenate([np.mean(data[key], 0) for key in self.resp_keys], axis=0)
            else:
                responses = np.concatenate([data[key] for key in self.resp_keys], axis=0)
            to_return_keys, to_return_vals = ["images", "responses"], [stimuli, responses]
            if self.stim_transform is not None:
                to_return_vals[0] = self.stim_transform(to_return_vals[0])
            if self.resp_transform is not None:
                to_return_vals[1] = self.resp_transform(to_return_vals[1])
            if self.clamp_neg_resp:
                to_return_vals[1].clamp_min_(0)
            if self.return_coords:
                to_return_keys.append("neuron_coords")
                to_return_vals.append(self.coords["all"])
            if self.return_ori:
                to_return_keys.append("neuron_ori")
                to_return_vals.append(self.ori["all"])
            return namedtuple("Datapoint", to_return_keys)(*to_return_vals)


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, cache=None):
        self.dataset = dataset
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            self.cache[idx] = self.dataset[idx]
            return self.cache[idx]

    def __getattr__(self, item):
        return getattr(self.dataset, item)

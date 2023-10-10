import os
from collections import OrderedDict, namedtuple
import random
import logging
import pickle

import numpy as np
import pandas as pd
import skimage.transform
import sklearn.model_selection
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

from csng.utils import RunningStats


def create_lurz_data_loaders(train_dataset, val_dataset, test_dataset, seed=1, test_with_repeats=True, batch_size=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_loaders = OrderedDict({
        "train": OrderedDict({"spiking": DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)}),
        "validation": OrderedDict({"spiking": DataLoader(val_dataset, 256)}),
        "test": OrderedDict({"spiking": DataLoader(test_dataset, batch_size=None if test_with_repeats else 256)})
    })

    return data_loaders


def create_test_dataset(inputs, targets):
    assert inputs.shape[0] == inputs.shape[0], "The number of samples in inputs and targets do not match."
    expanded_inputs = np.expand_dims(inputs, 1)
    repeated_inputs = np.repeat(expanded_inputs, targets.shape[1], 1)

    return repeated_inputs, targets


def unpickle_dataset(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
        return data_dict["stim"], data_dict["resp"], data_dict["sheets"]


def prepare_spiking_data_loaders(
        train_path,
        val_path,
        test_path,
        image_size,
        crop,
        batch_size=64,
    ):
    """
    Prepare "Lurz-style" dataloaders from spiking data Args: train_path: Path to the pickled train dataset test_path:
    Path to the pickled test dataset images_count: Limit of images used for training. Set to -1 for no limit.
    neurons_count: Limit of neurons to use. Set -1 for no limit. image_size: Integer, or (height, width). Target size
    of stimuli images. Set to -1 for no resize crop: Boolean. Whether to crop the stimuli images rather than resizing
    them. validation_size: Float or integer. Size of the validation subset created from the training data. Same
    semantics as `test_size` in `sklearn.model_selection.train_test_split`.

    Returns: Ordered dictionary with "train", "validation" and "test" entries. Each containing another dictionary
    with only one entry "spiking" that finally contains a data loader.

    """
    # train_stimuli, train_responses, sheets = unpickle_dataset(train_path)
    test_stimuli, test_responses, _ = unpickle_dataset(test_path)

    # Create dataset transforms
    if image_size != -1:
        print(f"Resize: {image_size}")
        if crop:
            transforms = [
                NumpyImageCrop(image_size),
                NumpyToTensor()
            ]
        else:
            transforms = [
                NumpyImageResize(image_size),
                NumpyToTensor()
            ]
        transform = torchvision.transforms.Compose(transforms)
    else:
        # train_stimuli = np.expand_dims(train_stimuli, 1)  # Channels first for pytorch
        test_stimuli = np.expand_dims(test_stimuli, 1)
        transform = NumpyToTensor()
        print("no resize")

    ### Create the datasets
    train_dataset = CachedDataset(PerSampleStoredDataset(
        dataset_dir=train_path,
        inputs_transform=transform,
        targets_transform="scale_by_std",
    ))
    val_dataset = CachedDataset(PerSampleStoredDataset(
        dataset_dir=val_path,
        inputs_transform=transform,
        targets_transform="scale_by_std",
        targets_std=train_dataset.targets_std,
    ))
    test_dataset = CachedDataset(NeuronNumPyDataset(
        inputs=test_stimuli,
        targets=test_responses,
        sheets=None,
        inputs_transform=transform,
        scale_targets=True, # does targets_transform="scale_by_std"
        targets_std=train_dataset.targets_std,
        with_repeats=True,
    ))

    print(f"Train dataset size: {len(train_dataset)}. Validation dataset size: {len(val_dataset)}. Test dataset size: {len(test_dataset)}.")

    data_loaders = create_lurz_data_loaders(train_dataset, val_dataset, test_dataset, test_with_repeats=True, batch_size=batch_size)

    return data_loaders


class PerSampleStoredDataset(Dataset):
    def __init__(self, dataset_dir, inputs_transform=None, targets_transform=None, targets_std=None, with_repeats=False):
        self.__dataset_dir = dataset_dir
        self.__inputs_transform = inputs_transform if inputs_transform is not None else NumpyToTensor()
        if targets_transform is None:
            self.__targets_transform = NumpyToTensor()
        elif targets_transform == "scale_by_std":
            self.__targets_transform = targets_transform
        else:
            self.__targets_transform = targets_transform
        self.__file_names = [f for f in os.listdir(self.__dataset_dir) if f.endswith(".pkl") or f.endswith(".pickle")]
        self.__datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])
        self.__with_repeats = with_repeats

        ### calculate statistics
        self.__targets_std = targets_std
        self.__dataset_stats = None
        self.__targets_std_filepath = "targets_std.pickle"
        
        ### uncomment below for this class to work (now commented out just for loading the pretrained encoder)
        if self.__targets_std is None and self.__targets_transform == "scale_by_std":
            ### check if saved statistics exist
            if os.path.exists(self.__targets_std_filepath):
                print("Loading targets std from file...")
                with open(self.__targets_std_filepath, "rb") as f:
                    self.__targets_std = pickle.load(f)
            else:
                self.__calculate_targets_std(save_to=self.__targets_std_filepath)
        
    def __calculate_targets_std(self, save_to=None):
        print("Calculating std of targets...")

        for f_i, f_name in enumerate(os.listdir(self.__dataset_dir)):
            if f_name.endswith(".pkl") or f_name.endswith(".pickle"):
                with open(os.path.join(self.__dataset_dir, f_name), "rb") as f:
                    data = pickle.load(f)
                    responses = data["resp"]
                    if self.__dataset_stats is None:
                        self.__dataset_stats = RunningStats(num_components=responses.shape[-1])

                    self.__dataset_stats.update(responses)

                if f_i % 2000 == 0:
                    print(f"\tProcessed {f_i} files.")

        self.__targets_std = self.__dataset_stats.get_std()

        ### save calculated statistics
        if self.__targets_std is not None and save_to is not None:
            with open(save_to, "wb") as f:
                pickle.dump(self.__targets_std, f)
            print(f"Saved targets std to {save_to}")     

    @property
    def stats(self):
        return self.__dataset_stats

    @property
    def targets_std(self):
        return self.__targets_std
    
    def __len__(self):
        return len(self.__file_names)

    def __getitem__(self, idx):
        f_name = self.__file_names[idx]
        with open(os.path.join(self.__dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)
            images = data["stim"]
            responses = data["resp"]
            if self.__inputs_transform is not None:
                images = self.__inputs_transform(images)
                if self.__with_repeats:
                    images = torch.repeat_interleave(
                        images.unsqueeze(0), responses.shape[0], 0
                    )
            if self.__targets_transform is not None:
                if self.__targets_transform == "scale_by_std":
                    responses = responses / self.__targets_std
                else:
                    responses = self.__targets_transform(responses)
            return self.__datapoint(images, responses)
    

class NeuronNumPyDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, sheets: pd.Series, inputs_transform=None,
                 targets_transform=None, scale_targets: bool = True, targets_std = None, device: str = "cuda", with_repeats: bool = False):
        self.__inputs = inputs.astype(np.float32)
        self.__targets = targets.astype(np.float32)
        self.__sheets = sheets
        self.__scale_targets = scale_targets
        self.__datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])
        self.__targets_mean = np.mean(self.__targets, axis=0)
        self.__device = device
        self.__inputs_transform = inputs_transform if inputs_transform is not None else NumpyToTensor(device=device)
        self.__targets_transform = targets_transform if targets_transform is not None else NumpyToTensor(device=device)

        targets_std = np.std(self.__targets, axis=0) if targets_std is None else targets_std
        targets_zero_std_idx = targets_std < 0.001
        targets_std[targets_zero_std_idx] = 1  # Set std to 1 where division by zero is possible
        self.__targets_std = targets_std
        self.__with_repeats = with_repeats

    def __len__(self):
        return self.__inputs.shape[0]

    def __getitem__(self, idx):
        # Apply transforms
        # If with_repeats, repeat the inputs, so it matches the number of targets

        return self.__datapoint(
            self.__inputs_transform(self.__inputs[idx]) if not self.__with_repeats else
            torch.repeat_interleave(torch.unsqueeze(self.__inputs_transform(self.__inputs[idx]), 0),
                                    self.__targets[idx].shape[0], 0),
            self.__targets_transform(
                (self.__targets[idx]) / self.__targets_std if self.__scale_targets else self.__targets[idx]
            )
        )

    @property
    def n_neurons(self):
        return self.__inputs.shape[1]

    @property
    def targets_std(self):
        return self.__targets_std

    @property
    def sheets(self):
        return self.__sheets


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, cache=None):
        self.__dataset = dataset
        self.__cache = cache if cache is not None else {}

    # noinspection PyUnresolvedReferences
    def __len__(self):
        return self.__dataset.__len__()

    def __getitem__(self, idx):
        if idx in self.__cache:
            return self.__cache[idx]
        else:
            self.__cache[idx] = self.__dataset[idx]
            return self.__cache[idx]

    def __getattr__(self, item):
        return getattr(self.__dataset, item)


class NumpyImageResize:

    def __init__(self, size):
        self.__size = size

    def __call__(self, img,  *args, **kwargs):
        img = np.squeeze(img)
        img = skimage.transform.resize(img, self.__size)
        img = np.expand_dims(img, 0)
        return img


class NumpyImageCrop:

    def __init__(self, size):
        self.__size = size

    def __call__(self, img,  *args, **kwargs):
        img = np.squeeze(img)
        assert img.shape[0] >= self.__size[0] and img.shape[1] >= self.__size[1], "Size of the crop must be smaller " \
                                                                                  "than the image's dimensions "
        horizontal_gap = int((img.shape[0] - self.__size[0]) / 2)
        vertical_gap = int((img.shape[1] - self.__size[1]) / 2)
        img = img[horizontal_gap:horizontal_gap + self.__size[0], vertical_gap:vertical_gap + self.__size[1]]
        img = np.expand_dims(img, 0)
        return img


class NumpyToTensor:

    def __init__(self, device="cuda"):
        self.__device = device

    def __call__(self, x, *args, **kwargs):
        return torch.from_numpy(x).to(self.__device)

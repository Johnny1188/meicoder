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
from collections import namedtuple


def prepare_v1_dataloaders(
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
        stim_keys=("stim",),
        resp_keys=("resp",),
    ):
    ### prepare transforms
    if image_size != -1 and crop:
        stim_transform = [
            NumpyImageCrop(image_size),
            NumpyToTensor()
        ]
    elif image_size != -1 and not crop:
        stim_transform = [
            NumpyImageResize(image_size),
            NumpyToTensor()
        ]
    else:
        stim_transform = [
            NumpyToTensor(),
            lambda x: np.expand_dims(x, 0)
        ]
    if stim_normalize_mean is not None and stim_normalize_std is not None:
        stim_transform.append(torchvision.transforms.Normalize(mean=stim_normalize_mean, std=stim_normalize_std))
    stim_transform = torchvision.transforms.Compose(stim_transform)

    resp_transform = [NumpyToTensor()]
    if resp_normalize_mean is not None and resp_normalize_std is not None:
        resp_transform.append(torchvision.transforms.Lambda(lambda x: (x - resp_normalize_mean) / resp_normalize_std))
    elif resp_normalize_mean is not None:
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
    )
    if cached:
        train_dataset = CachedDataset(train_dataset)
        val_dataset = CachedDataset(val_dataset)
        test_dataset = CachedDataset(test_dataset)


    print(f"Train dataset size: {len(train_dataset)}. Validation dataset size: {len(val_dataset)}. Test dataset size: {len(test_dataset)}.")

    ### create data loaders
    data_loaders = {
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

    return data_loaders


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
    ):
        self.dataset_dir = dataset_dir
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
        self.file_names = [
            f_name for f_name in os.listdir(self.dataset_dir)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ]
        self.stim_keys = stim_keys
        self.resp_keys = resp_keys
        self.average_over_repeats = average_over_repeats

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
                self.ori = {
                    "V1_Exc_L23": np.array(pos_ori_file["V1_Exc_L23"]["ori"]),
                    "V1_Inh_L23": np.array(pos_ori_file["V1_Inh_L23"]["ori"]),
                }
                self.ori["all"] = np.concatenate((self.ori["V1_Exc_L23"], self.ori["V1_Inh_L23"]), axis=0)

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
            if self.return_coords:
                to_return_keys.append("neuron_coords")
                to_return_vals.append(self.coords["all"])
            if self.return_ori:
                to_return_keys.append("neuron_ori")
                to_return_vals.append(self.ori["all"])
            return namedtuple("Datapoint", to_return_keys)(*to_return_vals)


class NeuronNumPyDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, sheets: pd.Series, stim_transform=None,
                 resp_transform=None, scale_targets: bool = True, targets_std = None, device: str = "cpu", with_repeats: bool = False):
        self.stims = inputs.astype(np.float32)
        self.resps = targets.astype(np.float32)
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor(device=device)
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor(device=device)

    def __len__(self):
        return self.stims.shape[0]

    def __getitem__(self, idx):
        stim = self.stim_transform(self.stims[idx])
        resp = self.resp_transform(self.resps[idx])
        return stim, resp


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
    def __init__(self, device="cpu", unsqueeze_dims=None):
        self.__unsqueeze_dims = unsqueeze_dims
        self.__device = device

    def __call__(self, x, *args, **kwargs):
        if self.__unsqueeze_dims is not None:
            x = np.expand_dims(x, self.__unsqueeze_dims)
        return torch.from_numpy(x).float().to(self.__device)


class SyntheticDataset(Dataset):
    """ Extracts patches from the given images and encodes them with a pretrained encoder ("on the fly"). """

    def __init__(
        self,
        data_dir,
        patch_size,
        overlap,
        expand_stim_for_encoder=False,
        stim_transform=None,
        resp_transform=None,
        pretrained_encoder_path=None,
        device="cpu",
    ):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.expand_stim_for_encoder = expand_stim_for_encoder
        self.stim_transform = stim_transform
        self.resp_transform = resp_transform
        self.device = device
        
        self.encoder = self._load_encoder(
            pretrained_encoder_path=pretrained_encoder_path
        )

    def _load_encoder(self, pretrained_encoder_path):
        """ Load pretrained encoder (predefined config) and return it. """
        print("Loading encoder...")
        
        from data_orig import prepare_spiking_data_loaders
        from lurz2020.models.models import se2d_fullgaussian2d

        ### config only for the encoder
        DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model")
        spiking_data_loaders_config = {
            "train_path": os.path.join(DATA_PATH, "datasets", "train"),
            "val_path": os.path.join(DATA_PATH, "datasets", "val"),
            "test_path": os.path.join(DATA_PATH, "orig", "raw", "test.pickle"),
            "image_size": [50, 50],
            "crop": False,
            "batch_size": 32,
        }
        encoder_config = {
            "init_mu_range": 0.55,
            "init_sigma": 0.4,
            "input_kern": 19,
            "hidden_kern": 17,
            "hidden_channels": 32,
            "gamma_input": 1.0,
            "gamma_readout": 2.439,
            "grid_mean_predictor": None,
            "layers": 5
        }

        ### encoder
        data_loaders = prepare_spiking_data_loaders(**spiking_data_loaders_config)
        encoder = se2d_fullgaussian2d(
            **encoder_config,
            dataloaders=data_loaders,
            seed=2,
        )
        del data_loaders

        ### load pretrained core
        pretrained_core = torch.load(
            pretrained_encoder_path,
            map_location=self.device,
        )
        encoder.load_state_dict(pretrained_core, strict=True)
        encoder.to(self.device)
        return encoder.eval()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        file_name = self.image_files[index]
        file_path = os.path.join(self.data_dir, file_name)
        image = np.load(file_path)

        patches = self.extract_patches(image)
        return patches

    def get_encoder(self):
        return self.encoder

    def _scale_for_encoder(self, patches):
        ### scale to 0-100
        p_min, p_max = patches.min(), patches.max()
        return (patches - p_min) / (p_max - p_min) * 100

    @torch.no_grad()
    def extract_patches(self, img):
        h, w = img.shape[-2:]
        patches = []
        patch_size = self.patch_size
        if self.expand_stim_for_encoder:
            patch_size = int(np.ceil(patch_size * 1.5)) # pad and then crop

        for y in range(0, h - patch_size + 1, patch_size - self.overlap):
            for x in range(0, w - patch_size + 1, patch_size - self.overlap):
                patch = img[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)

        patches = torch.from_numpy(np.stack(patches)).float().to(self.device)
        
        ### encode patches = get resps
        if self.expand_stim_for_encoder:
            patches_for_encoder = F.interpolate(patches, size=self.patch_size, mode="bilinear", align_corners=False)
            ### take only the center of the patch - the encoder's resps cover only the center part
            patches = patches[:, :, int(patch_size / 4):int(patch_size / 4) + self.patch_size,
                        int(patch_size / 4):int(patch_size / 4) + self.patch_size]
        else:
            patches_for_encoder = patches
        patches_for_encoder = self._scale_for_encoder(patches_for_encoder)
        if self.encoder is not None:
            resps = self.encoder(patches_for_encoder)

        if self.resp_transform is not None:
            resps = self.resp_transform(resps)

        if self.stim_transform is not None:
            patches = self.stim_transform(patches)

        return patches, resps


# dataloader that mixes patches from different images within a batch
class BatchPatchesDataLoader():
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader.__iter__():
            patches, resps = batch[0], batch[1]
            patches = patches.view(-1, *patches.shape[2:])
            resps = resps.view(-1, *resps.shape[2:])

            ### shuffle patch-resp pairs
            idx = torch.randperm(patches.shape[0])
            patches = patches[idx]
            resps = resps[idx].float()

            yield patches, resps



class MixedBatchLoader:
    """
    A dataloader that interleaves or mixes batches from multiple dataloaders.

    Args:
        dataloaders (list): A list of dataloaders to interleave or mix batches from.
        neuron_coords (dict): A dictionary containing the neuron coordinates for each dataloader (data_key as the key).
        mixing_strategy (str): The strategy to use for mixing batches. Must be one of ["sequential", "parallel_min", "parallel_max"].
        device (str): The device to load the batches onto. Default is "cpu".

    Raises:
        AssertionError: If mixing_strategy is not one of ["sequential", "parallel_min", "parallel_max"].

    Attributes:
        dataloader_iters (list): A list of iterators for each dataloader.
        neuron_coords (dict): A dictionary containing the neuron coordinates for each dataloader (data_key as the key).
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
    def __init__(
        self,
        dataloaders,
        neuron_coords=None,
        mixing_strategy="sequential",
        data_keys=None,
        return_data_key=True,
        return_pupil_center=True,
        return_neuron_coords=True,
        device="cpu"
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
        
        self.return_data_key = return_data_key
        self.return_pupil_center = return_pupil_center
        self.return_neuron_coords = return_neuron_coords
        
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

    def add_dataloader(self, dataloader, neuron_coords=None, data_key=None):
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

        if neuron_coords is not None:
            for _data_key, coords in neuron_coords.items():
                if _data_key in self.neuron_coords.keys() \
                    and not torch.equal(self.neuron_coords[_data_key], coords.to(self.device)):
                    print(f"[WARNING]: neuron_coords for data_key {_data_key} already exists and are not the same. Overwriting.")
                self.neuron_coords[_data_key] = coords.to(self.device)

    def _get_sequential(self):
        """
        Interleaves batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        to_return = dict()
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                datapoint = next(self.dataloader_iters[dl_idx]["dl"])
                stim, resp = datapoint.images, datapoint.responses
                to_return[self.dataloader_iters[dl_idx]["data_key"]] = [stim.to(self.device), resp.to(self.device)]
                if self.return_neuron_coords:
                    if self.neuron_coords is None:
                        _neuron_coords = datapoint.neuron_coords
                    else:
                        _neuron_coords = self.neuron_coords[self.dataloader_iters[dl_idx]["data_key"]]
                    to_return[self.dataloader_iters[dl_idx]["data_key"]].append(_neuron_coords.to(self.device))
                if self.return_pupil_center:
                    _pupil_center = datapoint.pupil_center
                    to_return[self.dataloader_iters[dl_idx]["data_key"]].append(_pupil_center.to(self.device))
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
        """
        Mixes batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        empty_dataloader_idxs = set()
        to_return = dict()
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                datapoint = next(dataloader_iter["dl"])
                _stim, _resp = datapoint.images, datapoint.responses
                to_return[dataloader_iter["data_key"]] = [_stim.to(self.device), _resp.to(self.device)]
                if self.return_neuron_coords:
                    if self.neuron_coords is None:
                        _neuron_coords = datapoint.neuron_coords
                    else:
                        _neuron_coords = self.neuron_coords[dataloader_iter["data_key"]]
                    to_return[dataloader_iter["data_key"]].append(_neuron_coords.to(self.device))
                if self.return_pupil_center:
                    _pupil_center = datapoint.pupil_center
                    to_return[dataloader_iter["data_key"]].append(_pupil_center.to(self.device))
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
            # new_dataloader_iters = []
            # for dl_idx, dataloader_iter in enumerate(self.dataloader_iters):
            #     if dl_idx not in empty_dataloader_idxs:
            #         new_dataloader_iters.append(dataloader_iter)
            # self.dataloader_iters = new_dataloader_iters
            for dl_idx_to_remove in empty_dataloader_idxs:
                del self.dataloader_iters[dl_idx_to_remove]
            self.n_dataloaders = len(self.dataloader_iters)

        return to_return

        # ### concatenate
        # stim = torch.cat(stim, dim=0)
        # resp = torch.cat(resp, dim=0)
        # if self.return_data_key:
        #     return stim, resp, None
        # return stim, resp

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

        if len(out) == 0: # no more data
            raise StopIteration

        return out

# class MixedBatchLoader:
#     """ Mixes batches from multiple dataloaders into one batch. """
#     def __init__(self, dataloaders, mixing_strategy="sequential", device="cpu"):
#         assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
#             f"mixing_strategy must be one of ['sequential', 'parallel'], but got {mixing_strategy}"

#         self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
#         self.n_dataloaders = len(self.dataloader_iters)
#         self.mixing_strategy = mixing_strategy
#         self.device = device
#         self.batch_idx = 0
        
#         if self.mixing_strategy == "sequential":
#             self.n_batches = sum([len(dataloader) for dataloader in dataloaders])
#         elif self.mixing_strategy == "parallel_max":
#             self.n_batches = max([len(dataloader) for dataloader in dataloaders])
#         elif self.mixing_strategy == "parallel_min":
#             self.n_batches = min([len(dataloader) for dataloader in dataloaders])

#     def _get_sequential(self):
#         ### interleave multiple dataloaders - one after another
#         while True:
#             try:
#                 stim, resp = next(self.dataloader_iters[self.batch_idx % self.n_dataloaders])
#                 break
#             except StopIteration:
#                 self.dataloader_iters.pop(self.batch_idx % self.n_dataloaders)
#                 self.n_dataloaders -= 1
#                 if self.n_dataloaders == 0:
#                     return None, None
#                 else:
#                     continue
#         return stim.to(self.device), resp.to(self.device)
    
#     def _get_parallel(self):
#         ### mix single batches from all dataloaders into one batch
#         stim, resp = [], []
#         empty_dataloader_idxs = set()
#         for d_idx, dataloader_iter in enumerate(self.dataloader_iters):
#             try:
#                 _stim, _resp = next(dataloader_iter)
#                 stim.append(_stim.to(self.device))
#                 resp.append(_resp.to(self.device))
#             except StopIteration:
#                 if self.mixing_strategy == "parallel_min":
#                     ### if a single dataloader ends, end the whole loop
#                     _ = [empty_dataloader_idxs.add(_d_idx) for _d_idx in range(0, self.n_dataloaders)]
#                 elif self.mixing_strategy == "parallel_max":
#                     ### if a single dataloader ends, continue with the remaining ones
#                     empty_dataloader_idxs.add(d_idx)
#                 else:
#                     raise NotImplementedError
        
#         ### remove empty dataloaders
#         if len(empty_dataloader_idxs) > 0:
#             new_dataloader_iters = []
#             for d_idx, dataloader_iter in enumerate(self.dataloader_iters):
#                 if d_idx not in empty_dataloader_idxs:
#                     new_dataloader_iters.append(dataloader_iter)
#             self.dataloader_iters = new_dataloader_iters
#             self.n_dataloaders = len(new_dataloader_iters)

#         if len(stim) == 0:
#             return None, None

#         ### concatenate
#         stim = torch.cat(stim, dim=0)
#         resp = torch.cat(resp, dim=0)
#         return stim, resp

#     def __len__(self):
#         return self.n_batches

#     def __iter__(self):
#         return self

#     def __next__(self):
#         self.batch_idx += 1
#         if self.mixing_strategy == "sequential":
#             stim, resp = self._get_sequential()
#         elif self.mixing_strategy in ("parallel_min", "parallel_max"):
#             stim, resp = self._get_parallel()
#         else:
#             raise NotImplementedError
        
#         if stim is None: # no more data
#             raise StopIteration
#         return stim, resp


"""
- Loading synthetic data within a script:
syn_data_imgs_path = os.path.join(os.environ["DATA_PATH"], "sensorium22", "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6", "data", "images")
resp_mean = torch.from_numpy(np.load(os.path.join(DATA_PATH, "responses_mean_from_syn_dataset.npy"))).float()
resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, "responses_std_from_syn_dataset.npy"))).float()

config["data"]["syn_data"] = {
    "dataset": {
        # "data_dir": syn_data_imgs_path,
        "patch_size": 50,
        "overlap": 0,
        "expand_stim_for_encoder": False,
        "stim_transform": transforms.Normalize(
            mean=114.457,
            std=51.356,
        ),
        "resp_transform": csng.utils.Normalize(
            mean=resp_mean.to(config["device"]),
            std=resp_std.to(config["device"]),
        ),
        "device": config["device"],
        "pretrained_encoder_path": os.path.join(
            DATA_PATH, "models", "spiking_scratch_tunecore_68Y_model.pth"
        ),
    },
    "dataloader": {
        "batch_size": 2,
        "shuffle": True,
    }
}

syn_datasets = {
    "train": SyntheticDataset(
        data_dir=os.path.join(DATA_PATH, "synthetic_data", "train"),
        **config["data"]["syn_data"]["dataset"],
    ),
    "val": SyntheticDataset(
        data_dir=os.path.join(DATA_PATH, "synthetic_data", "val"),
        **config["data"]["syn_data"]["dataset"],
    ),
    "test": SyntheticDataset(
        data_dir=os.path.join(DATA_PATH, "synthetic_data", "test"),
        **config["data"]["syn_data"]["dataset"],
    ),
}

syn_dataloaders = {
    "train": BatchPatchesDataLoader(DataLoader(
        dataset=syn_datasets["train"],
        **config["data"]["syn_data"]["dataloader"],
    )),
    "val": BatchPatchesDataLoader(DataLoader(
        dataset=syn_datasets["val"],
        **config["data"]["syn_data"]["dataloader"],
    )),
    "test": BatchPatchesDataLoader(DataLoader(
        dataset=syn_datasets["test"],
        **config["data"]["syn_data"]["dataloader"],
    )),
}
"""

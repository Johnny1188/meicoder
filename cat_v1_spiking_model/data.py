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


def prepare_v1_dataloaders(
        train_path,
        val_path,
        test_path,
        image_size,
        crop,
        batch_size=64,
        stim_normalize_mean=None,
        stim_normalize_std=None,
        resp_normalize_mean=None,
        resp_normalize_std=None,
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
    train_dataset = CachedDataset(PerSampleStoredDataset(
        dataset_dir=train_path,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
    ))
    val_dataset = CachedDataset(PerSampleStoredDataset(
        dataset_dir=val_path,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
    ))

    with open(test_path, "rb") as f: # a single file
        data_dict = pickle.load(f)
        test_stimuli = data_dict["stim"]
        test_responses = data_dict["resp"]
        sheets = data_dict["sheets"]
    test_dataset = CachedDataset(NeuronNumPyDataset(
        inputs=test_stimuli,
        targets=test_responses,
        sheets=sheets,
        stim_transform=stim_transform,
        resp_transform=resp_transform,
    ))

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
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    }

    return data_loaders


class PerSampleStoredDataset(Dataset):
    def __init__(self, dataset_dir, stim_transform=None, resp_transform=None):
        self.dataset_dir = dataset_dir
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
        self.file_names = [
            f_name for f_name in os.listdir(self.dataset_dir)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        with open(os.path.join(self.dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)
            stimuli = data["stim"]
            responses = data["resp"]
            if self.stim_transform is not None:
                stimuli = self.stim_transform(stimuli)
            if self.resp_transform is not None:
                responses = self.resp_transform(responses)
            return stimuli, responses


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
    """ Mixes batches from multiple dataloaders into one batch. """
    def __init__(self, dataloaders, mixing_strategy="sequential", device="cpu"):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel'], but got {mixing_strategy}"

        self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self.n_dataloaders = len(self.dataloader_iters)
        self.mixing_strategy = mixing_strategy
        self.device = device
        self.batch_idx = 0
        
        if self.mixing_strategy == "sequential":
            self.n_batches = sum([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min([len(dataloader) for dataloader in dataloaders])

    def _get_sequential(self):
        ### interleave multiple dataloaders - one after another
        while True:
            try:
                stim, resp = next(self.dataloader_iters[self.batch_idx % self.n_dataloaders])
                break
            except StopIteration:
                self.dataloader_iters.pop(self.batch_idx % self.n_dataloaders)
                self.n_dataloaders -= 1
                if self.n_dataloaders == 0:
                    return None, None
                else:
                    continue
        return stim.to(self.device), resp.to(self.device)
    
    def _get_parallel(self):
        ### mix single batches from all dataloaders into one batch
        stim, resp = [], []
        empty_dataloader_idxs = set()
        for d_idx, dataloader_iter in enumerate(self.dataloader_iters):
            try:
                _stim, _resp = next(dataloader_iter)
                stim.append(_stim.to(self.device))
                resp.append(_resp.to(self.device))
            except StopIteration:
                if self.mixing_strategy == "parallel_min":
                    ### if a single dataloader ends, end the whole loop
                    _ = [empty_dataloader_idxs.add(_d_idx) for _d_idx in range(0, self.n_dataloaders)]
                elif self.mixing_strategy == "parallel_max":
                    ### if a single dataloader ends, continue with the remaining ones
                    empty_dataloader_idxs.add(d_idx)
                else:
                    raise NotImplementedError
        
        ### remove empty dataloaders
        if len(empty_dataloader_idxs) > 0:
            new_dataloader_iters = []
            for d_idx, dataloader_iter in enumerate(self.dataloader_iters):
                if d_idx not in empty_dataloader_idxs:
                    new_dataloader_iters.append(dataloader_iter)
            self.dataloader_iters = new_dataloader_iters
            self.n_dataloaders = len(new_dataloader_iters)

        if len(stim) == 0:
            return None, None

        ### concatenate
        stim = torch.cat(stim, dim=0)
        resp = torch.cat(resp, dim=0)
        return stim, resp

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_idx += 1
        if self.mixing_strategy == "sequential":
            stim, resp = self._get_sequential()
        elif self.mixing_strategy in ("parallel_min", "parallel_max"):
            stim, resp = self._get_parallel()
        else:
            raise NotImplementedError
        
        if stim is None: # no more data
            raise StopIteration
        return stim, resp


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

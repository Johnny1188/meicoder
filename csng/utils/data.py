import os
import torch
import numpy as np
import skimage
from torch.utils.data import DataLoader, Dataset
import pickle
from collections import namedtuple
from pathlib import Path



def crop(x, win):
    ### win: (x1, x2, y1, y2) or (slice(x1, x2), slice(y1, y2) or (height, width) of middle patch
    if win == None:
        return x
    if isinstance(win[0], int) and len(win) == 4: # (x1, x2, y1, y2)
        if x.shape[-2] == win[1] - win[0] and x.shape[-1] == win[3] - win[2]:
            return x
        return x[..., win[0]:win[1], win[2]:win[3]]
    elif isinstance(win[0], int) and len(win) == 2: # (height, width) of middle patch
        if x.shape[-2] == win[0] and x.shape[-1] == win[1]:
            return x
        return x[..., (x.shape[-2] - win[0])//2:(x.shape[-2] + win[0])//2, (x.shape[-1] - win[1])//2:(x.shape[-1] + win[1])//2]
    else: # (slice(x1, x2), slice(y1, y2))
        if x.shape[-2] == win[0].stop - win[0].start and x.shape[-1] == win[1].stop - win[1].start:
            return x
        return x[..., win[0], win[1]]


def standardize(x, dim=(1,2,3), eps=1e-8, min=None, max=None, inplace=False):
    ### to [0,1] range
    x_min = x.amin(dim=dim, keepdim=True) if min is None else min
    x_max = x.amax(dim=dim, keepdim=True) if max is None else max

    if inplace:
        x -= x_min
        x /= (x_max - x_min + eps)
        return x
    else:
        return (x - x_min) / (x_max - x_min + eps)


def normalize(x, dim=(1,2,3), mean=None, std=None, inplace=False):
    ### mean 0, std 1
    x_mean = x.mean(dim=dim, keepdim=True) if mean is None else mean
    x_std = x.std(dim=dim, keepdim=True) if std is None else std

    if inplace:
        x -= x_mean
        x /= (x_std + 1e-8)
        return x
    else:
        return (x - x_mean) / (x_std + 1e-8)


class Normalize:
    """Class to normalize data."""

    def __init__(self, mean, std, center_data=True, clip_min=None, clip_max=None):
        self.mean = mean
        self.std = std
        self.center_data = center_data
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, values):
        if self.center_data:
            out = (values - self.mean) / (self.std + 1e-8)
        else:
            out = ((values - self.mean) / (self.std + 1e-8)) + self.mean

        if self.clip_min is not None or self.clip_max is not None:
            out = np.clip(out, self.clip_min, self.clip_max)

        return out


def get_mean_and_std(dataset=None, dataloader=None, verbose=False):
    """ Compute the mean and std value of dataset. """
    assert dataset is not None or dataloader is not None, "Either dataset or dataloader must be provided."
    if dataloader is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        data_len = len(dataset)
    else:
        data_len = len(dataloader.dataset)

    mean_inputs, std_inputs = torch.zeros(3), torch.zeros(3)
    mean_targets, std_targets = torch.zeros(1), torch.zeros(1)
    
    print('==> Computing mean and std..')
    # for inp_idx, (inputs, targets, _, _) in enumerate(dataloader):
    for inp_idx, (inputs, targets) in enumerate(dataloader):
        for c in range(inputs.size(1)):
            mean_inputs[c] += inputs[:,c,:,:].cpu().mean()
            std_inputs[c] += inputs[:,c,:,:].cpu().std()
        mean_targets += targets.cpu().mean()
        std_targets += targets.cpu().std()
        
        if verbose and inp_idx % 1000 == 0:
            print(f"Processed {inp_idx} / {data_len} samples.")
    mean_inputs.div_(len(dataset))
    std_inputs.div_(len(dataset))
    mean_targets.div_(len(dataset))
    std_targets.div_(len(dataset))
    
    return {
        "inputs": {
            "mean": mean_inputs,
            "std": std_inputs
        },
        "targets": {
            "mean": mean_targets,
            "std": std_targets
        }
    }


class MixedBatchLoader:
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
            assert len(data_keys) == len(dataloaders), \
                f"len(data_keys) must be equal to len(dataloaders), but got {len(data_keys)} and {len(dataloaders)}"
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


class NumpyToTensor:
    def __init__(self, device="cpu", unsqueeze_dims=None):
        self.unsqueeze_dims = unsqueeze_dims
        self.device = device

    def __call__(self, x, *args, **kwargs):
        if self.unsqueeze_dims is not None:
            x = np.expand_dims(x, self.unsqueeze_dims)
        return torch.from_numpy(x).float().to(self.device)


class NumpyImageResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img,  *args, **kwargs):
        img = np.squeeze(img)
        img = skimage.transform.resize(img, self.size)
        img = np.expand_dims(img, 0)
        return img


class NumpyImageCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img,  *args, **kwargs):
        img = np.squeeze(img)
        assert img.shape[0] >= self.size[0] and img.shape[1] >= self.size[1], \
            "Size of the crop must be smaller than the image's dimensions."
        horizontal_gap = int((img.shape[0] - self.size[0]) / 2)
        vertical_gap = int((img.shape[1] - self.size[1]) / 2)
        img = img[horizontal_gap:horizontal_gap + self.size[0], vertical_gap:vertical_gap + self.size[1]]
        img = np.expand_dims(img, 0)
        return img


class PerSampleStoredDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        stim_transform=None,
        resp_transform=None,
        additional_keys=None,
        clamp_neg_resp=False,
        avg_resp=True,
        dataset_shuffle_seed=None,
        device="cpu",
    ):
        self.dataset_dir = dataset_dir
        self.file_names = np.array([
            f_name for f_name in os.listdir(self.dataset_dir)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ])
        if dataset_shuffle_seed is None:
            self.file_names = np.sort(self.file_names)
        else:
            np.random.default_rng(dataset_shuffle_seed).shuffle(self.file_names)
        self.parent_dir = Path(self.dataset_dir).parent.absolute()
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor(device=device)
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor(device=device)
        self.additional_keys = additional_keys
        self.clamp_neg_resp = clamp_neg_resp
        self.avg_resp = avg_resp
        self.keys_to_return = ["images", "responses"]
        self.device = device
        if self.additional_keys is not None:
            self.keys_to_return.extend(self.additional_keys)

    def __len__(self):
        return len(self.file_names)

    @property
    def n_neurons(self):
        return self[0].responses.shape[-1]

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        with open(os.path.join(self.dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)
            vals_to_return = [data["stim"], data["resp"]]
            
            ### average responses
            if self.avg_resp:
                vals_to_return[1] = vals_to_return[1].mean(axis=0)

            ### transforms
            if self.stim_transform is not None:
                vals_to_return[0] = self.stim_transform(vals_to_return[0])
            if self.resp_transform is not None:
                vals_to_return[1] = self.resp_transform(vals_to_return[1])
            if self.clamp_neg_resp:
                vals_to_return[1].clamp_min_(0)

            ### additional keys
            if self.additional_keys is not None:
                for key in self.additional_keys:
                    vals_to_return.append(data[key])

            ### to device
            vals_to_return = [val.to(self.device) for val in vals_to_return]

            return namedtuple("Datapoint", self.keys_to_return)(*vals_to_return)

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pickle
import numpy as np
from collections import OrderedDict, namedtuple, defaultdict
from pathlib import Path
from csng.utils import crop, Normalize

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")


class NumpyToTensor:
    def __init__(self, device="cpu", unsqueeze_dims=None):
        self.unsqueeze_dims = unsqueeze_dims
        self.device = device

    def __call__(self, x, *args, **kwargs):
        if self.unsqueeze_dims is not None:
            x = np.expand_dims(x, self.unsqueeze_dims)
        return torch.from_numpy(x).float().to(self.device)


class MixedBatchLoaderV2:
    def __init__(
        self,
        dataloaders,
        neuron_coords=None,
        mixing_strategy="sequential",
        max_batches=None,
        data_keys=None,
        return_data_key=True,
        return_pupil_center=False,
        return_neuron_coords=False,
        device="cpu",
    ):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel_min', 'parallel_max'], but got {mixing_strategy}"

        self.dataloaders = dataloaders
        self.dataloader_iters = {dl_idx: {"dl": iter(dataloader)} for dl_idx, dataloader in enumerate(dataloaders)}
        self.data_keys = data_keys
        assert len(data_keys) == len(dataloaders), \
            f"len(data_keys) must be equal to len(dataloaders), but got {len(data_keys)} and {len(dataloaders)}"
        for dl_idx, data_key in zip(self.dataloader_iters.keys(), data_keys):
            self.dataloader_iters[dl_idx]["data_key"] = str(data_key)
        self.dataloaders_left = list(self.dataloader_iters.keys())
        self.n_dataloaders = len(self.dataloader_iters)
        self.mixing_strategy = mixing_strategy
        self.max_batches = max_batches
        
        self.neuron_coords = neuron_coords
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

        self.datasets = dict()
        for dl, data_key in zip(dataloaders, data_keys):
            self.datasets[str(data_key)] = dl.dataset

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


class PerSampleStoredDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        stim_transform=None,
        resp_transform=None,
        additional_keys=None,
        clamp_neg_resp=False,
        avg_resp=True,
        device="cpu",
    ):
        self.dataset_dir = dataset_dir
        self.file_names = np.array([
            f_name for f_name in os.listdir(self.dataset_dir)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ])
        self.parent_dir = Path(self.dataset_dir).parent.absolute()
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor(device=device)
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor(device=device)
        self.additional_keys = additional_keys
        self.clamp_neg_resp = clamp_neg_resp
        self.avg_resp = avg_resp

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        with open(os.path.join(self.dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)
            to_return_keys = ["images", "responses"]
            to_return_vals = [data["stim"], data["resp"]]
            
            ### average responses
            if self.avg_resp:
                to_return_vals[1] = to_return_vals[1].mean(axis=0)

            ### transforms
            if self.stim_transform is not None:
                to_return_vals[0] = self.stim_transform(to_return_vals[0])
            if self.resp_transform is not None:
                to_return_vals[1] = self.resp_transform(to_return_vals[1])
            if self.clamp_neg_resp:
                to_return_vals[1].clamp_min_(0)

            ### additional keys
            if self.additional_keys is not None:
                for key in self.additional_keys:
                    to_return_keys.append(key)
                    to_return_vals.append(data[key])

            return namedtuple("Datapoint", to_return_keys)(*to_return_vals)


def get_brainreader_data(config):
    dls = defaultdict(dict)
    for sess_id in config["sessions"]:
        ### prepare stim transforms
        stim_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((36, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.to(config["device"])),
        ])
        if config["normalize_stim"]:
            stim_mean = np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "stimuli_mean.npy")).item()
            stim_std = np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "stimuli_std.npy")).item()
            stim_transform.transforms.append(
                torchvision.transforms.Normalize(mean=stim_mean, std=stim_std)
            )

        ### prepare resp transforms
        resp_transform = torchvision.transforms.Compose([
            NumpyToTensor(device=config["device"]),
        ])
        if config["normalize_resp"]:
            resp_mean = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_mean.npy"))).to(config["device"])
            resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_std.npy"))).to(config["device"])
            resp_transform.transforms.append(
                Normalize(mean=resp_mean, std=resp_std)
            )
        elif config["div_resp_by_std"]:
            resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_std.npy"))).to(config["device"])
            resp_transform.transforms.append(
                Normalize(mean=0, std=resp_std)
            )

        ### setup dataloaders
        for data_part in ["train", "test", "val"]:
            dset = PerSampleStoredDataset(
                dataset_dir=os.path.join(config["data_dir"], str(sess_id), data_part),
                stim_transform=stim_transform,
                resp_transform=resp_transform,
                clamp_neg_resp=config["clamp_neg_resp"],
                additional_keys=config["additional_keys"],
                avg_resp=config["avg_test_resp"] if data_part == "test" else True,
            )
            dls[data_part][str(sess_id)] = DataLoader(
                dset,
                batch_size=config["batch_size"],
                shuffle=True if data_part == "train" else False,
            )
    
    dls_out = {
        "brainreader_mouse": {
            data_part: MixedBatchLoaderV2(
                dataloaders=[dls[data_part][str(sess_id)] for sess_id in config["sessions"]],
                neuron_coords=None,
                mixing_strategy=config["mixing_strategy"],
                max_batches=config["max_batches"],
                data_keys=list(dls[data_part].keys()),
                return_data_key=True,
                device=config["device"],
            ) for data_part in ["train", "test", "val"]
        }
    }

    return dls_out

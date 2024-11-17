import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pickle
import numpy as np
from collections import OrderedDict, namedtuple, defaultdict
from pathlib import Path

from csng.utils.data import crop, MixedBatchLoader, NumpyToTensor, Normalize


def get_brainreader_mouse_dataloaders(config):
    DATA_PATH = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
    dls = defaultdict(dict)
    for sess_id in config["sessions"]:
        ### prepare stim transforms
        stim_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
        ])
        # by default resize to 36x64
        if config.get("resize_stim_to", (36, 64)) is not None:
            stim_transform.transforms.append(
                torchvision.transforms.Resize(config.get("resize_stim_to", (36, 64)))
            )
        stim_transform.transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.to(config["device"])),
        ])
        # normalize stimuli
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
            data_part: MixedBatchLoader(
                dataloaders=[dls[data_part][str(sess_id)] for sess_id in config["sessions"]],
                neuron_coords=None,
                mixing_strategy=config["mixing_strategy"],
                max_batches=config["max_batches"],
                data_keys=list(dls[data_part].keys()),
                return_data_key=True,
                return_pupil_center=False,
                return_neuron_coords=False,
                device=config["device"],
            ) for data_part in ["train", "test", "val"]
        }
    }

    return dls_out


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

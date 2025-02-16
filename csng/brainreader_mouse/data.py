import os
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from collections import defaultdict

from csng.utils.data import MixedBatchLoader, NumpyToTensor, Normalize, PerSampleStoredDataset


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
                dataset_shuffle_seed=config.get(f"{data_part}_dataset_shuffle_seed", None),
            )
            dls[data_part][str(sess_id)] = DataLoader(
                dset,
                batch_size=config["batch_size"],
                shuffle=True if data_part == "train" else False,
                drop_last=config.get("drop_last", False),
            )
    
    dls_out = {
        "brainreader_mouse": {
            data_part: MixedBatchLoader(
                dataloaders=[dls[data_part][str(sess_id)] for sess_id in config["sessions"]],
                neuron_coords=config.get("neuron_coords", None), # ground truth data doesn't have neuron coords
                mixing_strategy=config["mixing_strategy"],
                max_batches=config["max_batches"] if f"max_{data_part}_batches" not in config else config[f"max_{data_part}_batches"],
                data_keys=list(dls[data_part].keys()),
                return_data_key=True,
                return_pupil_center=False,
                return_neuron_coords="neuron_coords" in config,
                device=config["device"],
            ) for data_part in ["train", "test", "val"]
        }
    }

    return dls_out

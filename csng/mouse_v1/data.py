import os
import pickle
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from nnfabrik.builder import get_data
from collections import namedtuple

import csng
from csng.utils.data import MixedBatchLoader, NumpyToTensor, PerSampleStoredDataset


def get_mouse_v1_dataloaders(config):
    ### get dataloaders
    _dls = get_data(config["data"]["mouse_v1"]["dataset_fn"], config["data"]["mouse_v1"]["dataset_config"])

    if config["data"]["mouse_v1"]["average_test_multitrial"]:
        _dls["test"] = average_test_multitrial(_dls["test"], config["data"])

    dls = {
        "mouse_v1": {
            "train": MixedBatchLoader(
                dataloaders=[_dls["train"][data_key] for data_key in _dls["train"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["data"]["mixing_strategy"],
                max_batches=config["data"].get("max_training_batches"),
                data_keys=list(_dls["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ) if config["data"]["mouse_v1"]["skip_train"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ),
            "val": MixedBatchLoader(
                dataloaders=[_dls["validation"][data_key] for data_key in _dls["validation"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=list(_dls["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ) if config["data"]["mouse_v1"]["skip_val"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ),
            "test": MixedBatchLoader(
                dataloaders=[_dls["test"][data_key] for data_key in _dls["test"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=list(_dls["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ) if config["data"]["mouse_v1"]["skip_test"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            ),
            "test_no_resp": MixedBatchLoader(
                dataloaders=[_dls["final_test"][data_key] for data_key in _dls["final_test"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["data"]["mixing_strategy"],
                data_keys=list(_dls["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                return_neuron_coords=True,
                device=config["data"]["mouse_v1"]["device"],
            )
        }
    }
    
    ### get cell coordinates
    neuron_coords = {
        data_key: torch.tensor(d.neurons.cell_motor_coordinates, dtype=torch.float32, device=config["data"]["mouse_v1"]["device"])
        for data_key, d in zip(list(_dls["train"].keys()), [_dl.dataset for _dl in _dls["train"].values()])
    }
    if config["data"]["mouse_v1"]["normalize_neuron_coords"]: # normalize coordinates to [-1, 1]
        for data_key in neuron_coords.keys():
            ### normalize x,y,z separately
            for dim_idx in range(neuron_coords[data_key].shape[-1]):
                neuron_coords[data_key][:, dim_idx] = \
                    (neuron_coords[data_key][:, dim_idx] - neuron_coords[data_key][:, dim_idx].min()) \
                    / (neuron_coords[data_key][:, dim_idx].max() - neuron_coords[data_key][:, dim_idx].min()) * 2 - 1

    ### assign neuron_coords to dataloaders
    for tier in ["train", "val", "test", "test_no_resp"]:
        dls["mouse_v1"][tier].neuron_coords = neuron_coords

    return dls, neuron_coords


def average_test_multitrial(dataloaders, config):
    for data_key, dataloader in dataloaders.items():
        ### check if the averaged test dataset has been already created and saved
        if os.path.exists(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl")):
            # print(f"[INFO] Loading averaged test dataset for {data_key}...")
            with open(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl"), "rb") as f:
                averaged_test_dataset = pickle.load(f)
            dataloaders[data_key] = DataLoader(
                SamplesDataset(
                    stims=torch.from_numpy(averaged_test_dataset["stims"]),
                    resps=torch.from_numpy(averaged_test_dataset["resps"]),
                    pupil_centers=torch.from_numpy(averaged_test_dataset["pupil_centers"]),
                    stim_transform=lambda x: x,
                    resp_transform=lambda x: x,
                    device=config["mouse_v1"]["device"],
                ),
                batch_size=config["mouse_v1"].get("test_batch_size", config["mouse_v1"]["dataset_config"]["batch_size"]),
                shuffle=False,
            )
            continue
        else:
            print(f"[INFO] Averaging test dataset for {data_key}...")
            ### get the responses, stimuli, pupil centers and image ids from the test set
            stims, resps, p_centers, img_ids = get_multitrial_info(dataloader.dataset, tier="test")
            repeats_stims = split_on_idx(stims, img_ids)
            repeats_resps = split_on_idx(resps, img_ids)
            repeats_p_centers = split_on_idx(p_centers, img_ids)
            
            ### take a single stim from each multi-trials (they are the same),
            ### average the responses and pupil centers
            for sample_idx in range(len(repeats_stims)):
                for i in range(1, len(repeats_stims[sample_idx]) - 1):
                    assert np.array_equal(repeats_stims[sample_idx][i], repeats_stims[sample_idx][i + 1]), \
                        "Stimuli are not the same across repeats"
            stims = np.stack([repeats_stims[i][0] for i in range(len(repeats_stims))])
            stims = stims[:, np.newaxis, ...]
            resps = np.stack([repeats_resps[i].mean(0) for i in range(len(repeats_resps))])
            p_centers = np.stack([repeats_p_centers[i].mean(0) for i in range(len(repeats_p_centers))])
            dataloaders[data_key] = DataLoader(
                SamplesDataset(
                    stims=torch.from_numpy(stims),
                    resps=torch.from_numpy(resps),
                    pupil_centers=torch.from_numpy(p_centers),
                    stim_transform=lambda x: x,
                    resp_transform=lambda x: x,
                    device=config["mouse_v1"]["device"],
                ),
                batch_size=config["mouse_v1"]["dataset_config"]["batch_size"],
                shuffle=False,
            )

            if config["mouse_v1"]["save_test_multitrial"]:
                print(f"[INFO] Saving averaged test dataset for {data_key}...")
                with open(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl"), "wb") as f:
                    pickle.dump({
                        "stims": stims,
                        "resps": resps,
                        "pupil_centers": p_centers,
                    }, f)

    return dataloaders


def split_on_idx(x, ids):
    per_image_repeats = []
    for image_id in np.unique(ids):
        responses_across_repeats = x[ids == image_id]
        per_image_repeats.append(responses_across_repeats)

    return per_image_repeats


def get_multitrial_info(dataset, tier="test"):
    tiers = dataset.trial_info.tiers
    complete_image_ids = dataset.trial_info.frame_image_id

    responses, stimuli, pupil_centers, image_ids = [], [], [], []
    for i, datapoint in enumerate(dataset):
        if tiers[i] != tier:
            continue

        responses.append(datapoint.responses.cpu().numpy().squeeze())
        stimuli.append(datapoint.images.cpu().numpy().squeeze())
        pupil_centers.append(datapoint.pupil_center.cpu().numpy().squeeze())
        image_ids.append(complete_image_ids[i])

    responses = np.stack(responses)
    stimuli = np.stack(stimuli)
    pupil_centers = np.stack(pupil_centers)

    return stimuli, responses, pupil_centers, image_ids


def append_syn_dataloaders(dataloaders, config):
    DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
    for data_key in config["data_keys"]:
        ### divide by the per neuron std if the std is greater than 1% of the mean std (to avoid division by 0)
        if config.get("responses_shift_mean", True):
            resp_mean = torch.from_numpy(np.load(os.path.join(DATA_PATH, config["dir_name"], data_key, f"responses_mean_original.npy"))).float()
        else:
            resp_mean = 0
        resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, config["dir_name"], data_key, f"responses_std_original.npy"))).float()
        div_by = resp_std.clone()
        thres = 0.01 * resp_std.mean()
        idx = resp_std <= thres
        div_by[idx] = thres

        data_key_to_add = data_key \
            if (config["data_key_prefix"] is None or config["data_key_prefix"] == "") \
            else f"{config['data_key_prefix']}_{data_key}"
        neuron_coords = {
            data_key_to_add: torch.from_numpy(np.load(
                os.path.join(DATA_PATH, config["dir_name"], data_key, f"neuron_coords.npy")
            )).float()
        }

        for data_part in config["append_data_parts"]:
            dataloader = DataLoader(
                PerSampleStoredDataset(
                    dataset_dir=os.path.join(DATA_PATH, config["dir_name"], data_key, data_part),
                    stim_transform=lambda x: x,
                    resp_transform=csng.utils.Normalize(
                        mean=resp_mean,
                        std=div_by,
                        center_data=False, # keep the same mean, just scale
                        clip_min=config.get("responses_clip_min", None),
                        clip_max=config.get("responses_clip_max", None),
                    ),
                    device=config.get("device", "cpu"),
                ),
                batch_size=config["batch_size"],
                shuffle=False,
            )
            dataloaders["mouse_v1"][data_part].add_dataloader(
                dataloader,
                neuron_coords=neuron_coords,
                data_key=data_key_to_add,
            )
    
    return dataloaders


def append_data_aug_dataloaders(dataloaders, config):
    for data_transform in config["data_transforms"]:
        for data_part in config["append_data_parts"]:
            ### copy all base dataloaders
            curr_dl = dataloaders["mouse_v1"][data_part]
            new_dls = deepcopy(curr_dl.dataloaders)
            new_neuron_coords = deepcopy(curr_dl.neuron_coords)
            new_data_keys = deepcopy(curr_dl.data_keys)
            assert len(new_dls) == len(new_data_keys)

            if config["force_same_order"]:
                ### set the same generator state for samplers in the old and new dataloaders
                for old_dl, new_dl in zip(curr_dl.dataloaders, new_dls):
                    old_dl.sampler.generator = torch.Generator().manual_seed(config["seed"])
                    new_dl.sampler.generator = torch.Generator().set_state(old_dl.sampler.generator.get_state())

            ### wrap base dataloaders with dataloaders applying the transforms
            for dl_i, (dl, data_key) in enumerate(zip(new_dls, new_data_keys)):
                new_dl = DataloaderWrapper(
                    dataloader=dl,
                    transform=data_transform[data_key] if type(data_transform) == dict else data_transform[dl_i],
                )
                curr_dl.add_dataloader(
                    new_dl,
                    neuron_coords={data_key: new_neuron_coords[data_key]},
                    data_key=data_key,
                )

    return dataloaders


class DataloaderWrapper:
    def __init__(self, dataloader, transform):
        self.dataloader = dataloader
        self.transform = transform

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __iter__(self):
        for data in self.dataloader:
            yield self.transform(data)

    def __len__(self):
        return len(self.dataloader)


class SamplesDataset(Dataset):
    def __init__(self, stims, resps, pupil_centers=None, stim_transform=None, resp_transform=None, device="cpu"):
        self.stims = stims
        self.resps = resps
        self.pupil_centers = pupil_centers
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
        self.device = device

    def __len__(self):
        return len(self.stims)

    def __getitem__(self, idx):
        stimuli = self.stims[idx]
        responses = self.resps[idx]
        if self.stim_transform is not None:
            stimuli = self.stim_transform(stimuli)
        if self.resp_transform is not None:
            responses = self.resp_transform(responses)

        if self.pupil_centers is None:
            return namedtuple("Datapoint", ["images", "responses"])(stimuli.to(self.device), responses.to(self.device))
        else:
            return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(stimuli.to(self.device), responses.to(self.device), self.pupil_centers[idx].to(self.device))


# class PerSampleStoredDataset(Dataset):
#     def __init__(self, dataset_dir, stim_transform=None, resp_transform=None, device="cpu"):
#         self.dirname = dataset_dir
#         self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
#         self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
#         self.file_names = [
#             f_name for f_name in os.listdir(self.dirname)
#             if f_name.endswith(".pkl") or f_name.endswith(".pickle")
#         ]
#         self.device = device

#     @property
#     def n_neurons(self):
#         return self[0][1].shape[-1]

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         f_name = self.file_names[idx]
#         with open(os.path.join(self.dirname, f_name), "rb") as f:
#             data = pickle.load(f)
#             stimuli = data["stim"]
#             responses = data["resp"]
#             if self.stim_transform is not None:
#                 stimuli = self.stim_transform(stimuli)
#             if self.resp_transform is not None:
#                 responses = self.resp_transform(responses)

#             if "pupil_center" in data:
#                 return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(
#                     stimuli.to(self.device),
#                     responses.to(self.device),
#                     data["pupil_center"].to(self.device)
#                 )
#             else:
#                 return namedtuple("Datapoint", ["images", "responses"])(
#                     stimuli.to(self.device),
#                     responses.to(self.device)
#                 )

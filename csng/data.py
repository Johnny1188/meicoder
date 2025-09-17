import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from csng.utils.data import MixedBatchLoader, PerSampleStoredDataset, Normalize
from csng.brainreader_mouse.data import get_brainreader_mouse_dataloaders
from csng.mouse_v1.data import get_mouse_v1_dataloaders
from csng.cat_v1.data import get_cat_v1_dataloaders
from csng.allen.data import get_allen_dataloaders


def get_sample_data(dls, config, sample_from_tier="val"):
    s = {"stim": None, "resp": None, "sample_data_key": None, "sample_dataset": None}

    if "brainreader_mouse" in config["data"]:
        s["b_sample_dataset"] = "brainreader_mouse"
        b_dp = next(iter(dls[sample_from_tier][s["b_sample_dataset"]]))
        s["b_stim"], s["b_resp"], s["b_sample_data_key"] = b_dp[0]["stim"], b_dp[0]["resp"], b_dp[0]["data_key"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["b_stim"], s["b_resp"], s["b_sample_data_key"], s["b_sample_dataset"]
    if "cat_v1" in config["data"]:
        s["c_sample_dataset"] = "cat_v1"
        c_dp = next(iter(dls[sample_from_tier][s["c_sample_dataset"]]))
        s["c_stim"], s["c_resp"], s["c_sample_data_key"] = c_dp[0]["stim"], c_dp[0]["resp"], c_dp[0]["data_key"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["c_stim"], s["c_resp"], s["c_sample_data_key"], s["c_sample_dataset"]
    if "mouse_v1" in config["data"]:
        s["m_sample_dataset"] = "mouse_v1"
        m_dp = next(iter(dls[sample_from_tier][s["m_sample_dataset"]]))
        s["m_stim"], s["m_resp"], s["m_sample_data_key"], s["m_pupil_center"] = m_dp[0]["stim"], m_dp[0]["resp"], m_dp[0]["data_key"], m_dp[0]["pupil_center"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["m_stim"], s["m_resp"], s["m_sample_data_key"], s["m_sample_dataset"]
    if "allen" in config["data"]:
        s["a_sample_dataset"] = "allen"
        a_dp = next(iter(dls[sample_from_tier][s["a_sample_dataset"]]))
        s["a_stim"], s["a_resp"], s["a_sample_data_key"] = a_dp[0]["stim"], a_dp[0]["resp"], a_dp[0]["data_key"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["a_stim"], s["a_resp"], s["a_sample_data_key"], s["a_sample_dataset"]

    return s


def get_dataloaders(config):
    dls = dict(train=dict(), val=dict(), test=dict())
    neuron_coords = dict()

    ### brainreader mouse
    if "brainreader_mouse" in config["data"]:
        _dls = get_brainreader_mouse_dataloaders(config=config["data"]["brainreader_mouse"])

        ### add to data loaders
        for tier in ("train", "val", "test"):
            dls[tier]["brainreader_mouse"] = _dls["brainreader_mouse"][tier]
        if _dls["brainreader_mouse"]["train"].neuron_coords is None:
            neuron_coords["brainreader_mouse"] = {data_key: None for data_key in _dls["brainreader_mouse"]["train"].data_keys}
        else:
            neuron_coords["brainreader_mouse"] = _dls["brainreader_mouse"]["train"].neuron_coords

    ### mouse v1 - base
    if "mouse_v1" in config["data"] and config["data"]["mouse_v1"] is not None:
        m_dls, _neuron_coords = get_mouse_v1_dataloaders(config=config)

        ### add to data loaders
        for tier in ("train", "val", "test"):
            dls[tier]["mouse_v1"] = m_dls["mouse_v1"][tier]
        neuron_coords["mouse_v1"] = _neuron_coords

    ### cat v1
    if "cat_v1" in config["data"]:
        c_dls = get_cat_v1_dataloaders(**config["data"]["cat_v1"]["dataset_config"])

        ### get neuron coordinates
        torch.allclose(c_dls["train"].dataset[0].neuron_coords, c_dls["train"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["train"].dataset[-1].neuron_coords, c_dls["val"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[0].neuron_coords, c_dls["val"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[-1].neuron_coords, c_dls["test"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["test"].dataset[0].neuron_coords, c_dls["test"].dataset[-1].neuron_coords), \
            "Neuron coordinates must be the same for all samples in the dataset"
        if config["data"]["cat_v1"].get("neuron_coords_to_use", None) is not None: # uses neuron coordinates from the config
            neuron_coords["cat_v1"] = {"cat_v1": config["data"]["cat_v1"]["neuron_coords_to_use"].float().to(config["device"])}
        else: # uses neuron coordinates from the dataset
            neuron_coords["cat_v1"] = {"cat_v1": c_dls["train"].dataset[0].neuron_coords.float().to(config["device"])}

        ### add to data loaders
        for tier in ("train", "val", "test"):
            dls[tier]["cat_v1"] = MixedBatchLoader(
                dataloaders=[c_dls[tier]],
                neuron_coords=neuron_coords["cat_v1"],
                mixing_strategy=config["data"]["mixing_strategy"],
                max_batches=config["data"].get("max_training_batches"),
                data_keys=["cat_v1"],
                neuron_idxs_to_use=config["data"]["cat_v1"].get("neuron_idxs_to_use", None),
                return_data_key=True,
                return_pupil_center=False, # no pupil center in cat_v1
                return_neuron_coords=True,
                device=config["device"],
            )

    ### Allen Visual Coding—Neuropixels dataset
    if "allen" in config["data"]:
        a_dls = get_allen_dataloaders(config=config["data"]["allen"])
        neuron_coords["allen"] = None

        ### add to data loaders
        for tier in ("train", "val", "test"):
            # dls[tier]["allen"] = a_dls["allen"][tier]
            dls[tier]["allen"] = MixedBatchLoader(
                dataloaders=[a_dls["allen"][tier]],
                neuron_coords=neuron_coords["allen"],
                mixing_strategy=config["data"]["allen"].get("mixing_strategy", "sequential"),
                max_batches=config["data"]["allen"].get("max_training_batches"),
                data_keys=["allen"],
                return_data_key=True,
                return_pupil_center=False, # no pupil center in allen
                return_neuron_coords=False,
                device=config["device"],
            )

    ### synthetic data
    if "syn_data" in config["data"]:
        s_dls, _neuron_coords = get_syn_dataloaders(config=config["data"]["syn_data"])

        ### add to data loaders
        for tier in config["data"]["syn_data"]["append_data_tiers"]:
            dls[tier]["syn_data"] = s_dls[tier]
        neuron_coords["syn_data"] = _neuron_coords

    return dls, neuron_coords


def get_syn_dataloaders(config):
    dls = {data_tier: [] for data_tier in config["append_data_tiers"]}
    neuron_coords = dict()
    for data_dict in config["data_dicts"]:
        ### divide by the per neuron std if the std is greater than 1% of the mean std (to avoid division by 0) - used by neuralpredictors
        resp_std = torch.from_numpy(np.load(os.path.join(data_dict["path"], "train", "stats", f"responses_std.npy"))).float()
        div_by = resp_std.clone()
        thres = 0.01 * resp_std.mean()
        idx = resp_std <= thres
        div_by[idx] = thres

        ### load mean for shifting the responses
        if config["responses_shift_mean"]:
            resp_mean = torch.from_numpy(np.load(os.path.join(data_dict["path"], "train", "stats", f"responses_mean.npy"))).float()
        else:
            resp_mean = 0

        ### load neuron coordinates
        if data_dict["load_neuron_coords"]:
            neuron_coords[data_dict["data_key"]] = torch.from_numpy(np.load(
                os.path.join(data_dict["path"], f"neuron_coords.npy"))
            ).float()

        ### append all data parts
        for data_tier in config["append_data_tiers"]:
            dls[data_tier].append(DataLoader(
                PerSampleStoredDataset(
                    dataset_dir=os.path.join(data_dict["path"], data_tier),
                    stim_transform=lambda x: x,
                    resp_transform=Normalize(
                        mean=resp_mean,
                        std=div_by,
                        center_data=False, # keep the same mean, just scale
                        clip_min=config.get("responses_clip_min", None),
                        clip_max=config.get("responses_clip_max", None),
                    ),
                    device=config.get("device", "cpu"),
                ),
                batch_size=config["batch_size"],
                shuffle=config["shuffle"],
            ))

    ### combine all dataloaders
    for data_tier in config["append_data_tiers"]:
        dls[data_tier] = MixedBatchLoader(
            dataloaders=dls[data_tier],
            neuron_coords=neuron_coords,
            mixing_strategy=config["mixing_strategy"],
            max_batches=config.get("max_training_batches", None),
            data_keys=[data_dict["data_key"] for data_dict in config["data_dicts"]],
            return_data_key=True,
            return_pupil_center=config["return_pupil_center"],
            return_neuron_coords=config["return_neuron_coords"],
            device=config["device"],
        )

    return dls, neuron_coords

import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import RunningStats, crop, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, plot_losses

from encoder import get_encoder
from data_utils import (
    get_mouse_v1_data,
    append_syn_dataloaders,
    append_data_aug_dataloaders,
    RespGaussianNoise,
)

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
print(f"{DATA_PATH=}")



class SyntheticDataset(Dataset):
    """ Extracts patches from the given images and encodes them with a pretrained encoder ("on the fly"). """
    def __init__(
        self,
        base_dl,
        data_key,
        patch_shape,
        overlap=(0, 0),
        stim_transform=None,
        resp_transform=None,
        device="cpu",
    ):
        assert base_dl.batch_size == 1, "Batch size must be 1."
        assert len(overlap) == 2, "Overlap must be a tuple of 2 elements."
        self.base_dl = base_dl
        self.base_dl_iter = iter(base_dl)
        self.data_key = data_key
        self.patch_shape = patch_shape
        self.overlap = overlap
        self.stim_transform = stim_transform
        self.resp_transform = resp_transform
        self.device = device

        self.encoder = self._load_encoder()

        self.seen_idxs = set()

    def _load_encoder(self):
        """ Load pretrained encoder (predefined config) and return it. """
        print("Loading encoder...")
        encoder = get_encoder(ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall.pth"), device=self.device, eval_mode=True)
        return encoder.eval()

    def __len__(self):
        return len(self.base_dl)

    def __getitem__(self, idx):
        if idx in self.seen_idxs:
            raise ValueError(f"Index {idx} already seen.")
        self.seen_idxs.add(idx)

        sample = next(self.base_dl_iter)
        img, pupil_center = sample.images.to(self.device), sample.pupil_center.to(self.device)

        patches, syn_resps = self.extract_patches(img=img, pupil_center=pupil_center)
        return patches, syn_resps, pupil_center.repeat(patches.shape[0], 1)

    # def _scale_for_encoder(self, patches):
    #     ### scale to 0-100
    #     p_min, p_max = patches.min(), patches.max()
    #     return (patches - p_min) / (p_max - p_min) * 100

    @torch.no_grad()
    def extract_patches(self, img, pupil_center=None):
        h, w = img.shape[-2:]
        patches = []
        patch_shape = self.patch_shape

        for y in range(0, h - patch_shape[0] + 1, patch_shape[0] - self.overlap[0]):
            for x in range(0, w - patch_shape[1] + 1, patch_shape[1] - self.overlap[1]):
                # patch = img[:, y:y+patch_size, x:x+patch_size]
                patch = img[..., y:y+patch_shape[0], x:x+patch_shape[1]]
                patches.append(patch)

        # patches = torch.from_numpy(np.stack(patches)).float().to(self.device)
        # patches = torch.stack(patches).float().to(self.device)
        ### merge into a batch
        patches = torch.cat(patches, dim=0).to(self.device)

        ### encode patches = get resps
        # if self.expand_stim_for_encoder:
        #     patches_for_encoder = F.interpolate(patches, size=self.patch_size, mode="bilinear", align_corners=False)
        #     ### take only the center of the patch - the encoder's resps cover only the center part
        #     patches = patches[:, :, int(patch_size / 4):int(patch_size / 4) + self.patch_size,
        #                 int(patch_size / 4):int(patch_size / 4) + self.patch_size]
        # patches = self._scale_for_encoder(patches)
        if self.encoder is not None:
            if hasattr(self.encoder, "shifter") and self.encoder.shifter is not None:
                resps = self.encoder(patches, data_key=self.data_key, pupil_center=pupil_center.expand(patches.shape[0], -1))
            else:
                resps = self.encoder(patches, data_key=self.data_key)

        if self.resp_transform is not None:
            resps = self.resp_transform(resps)

        if self.stim_transform is not None:
            patches = self.stim_transform(patches)

        return patches, resps
    
class BatchPatchesDataLoader():
    # dataloader that mixes patches from different images within a batch
    def __init__(self, dataloader):
        self.dataloader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader_iter)

    def __iter__(self):
        for batch in self.dataloader_iter:
            patches, resps, pupil_center = batch  # (B, N_patches, C, H, W)
            patches = patches.view(-1, *patches.shape[2:])
            resps = resps.view(-1, *resps.shape[2:])
            pupil_center = pupil_center.view(-1, *pupil_center.shape[2:])

            ### shuffle patch-resp pairs
            idx = torch.randperm(patches.shape[0])
            patches = patches[idx]
            resps = resps[idx].float()
            pupil_center = pupil_center[idx]

            yield patches, resps, pupil_center



if __name__ == "__main__":
    config = {
        "data": {
            "mixing_strategy": "sequential", # needed only with multiple base dataloaders
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0,
        "crop_win": (22, 36),
    }
    print(f"... Running on {config['device']} ...")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### get encoder
    encoder = get_encoder(ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall.pth"), device=config["device"], eval_mode=True)

    ### data
    dataloaders = dict()
    filenames = [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
        # "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # mouse 1
        # "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # sensorium+ (mouse 2)
        "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 3)
        # "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 4)
        # "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 5)
        # "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 6)
        # "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 7)
    ]
    for f_idx, f_name in enumerate(filenames):
        filenames[f_idx] = os.path.join(DATA_PATH, f_name)

    config["data"]["mouse_v1"] = {
        "paths": filenames,
        "dataset_fn": "sensorium.datasets.static_loaders",
        "dataset_config": {
            "paths": filenames,
            "normalize": True,
            "scale": 1.0, # 256x144 -> 64x36
            "include_behavior": False,
            "add_behavior_as_channels": False,
            "include_eye_position": True,
            "exclude": None,
            "file_tree": True,
            "cuda": False,
            "batch_size": 1,
            "seed": config["seed"],
            "use_cache": False,
        },
        "skip_train": False,
        "skip_val": False,
        "skip_test": False,
        "normalize_neuron_coords": True,
        "average_test_multitrial": True,
        "save_test_multitrial": True,
        "test_batch_size": 1,
        "device": config["device"],
    }
    ## get dataloaders and cell coordinates
    dataloaders, neuron_coords = get_mouse_v1_data(config["data"])

    syn_data_config = {
        "data_part": "test",
        "save_stats": False,
        "max_samples": 50000, # None or int
        "patch_dataset": {
            "data_key": None, # to be set
            "patch_shape": (36, 64),
            "overlap": (0, 0),
            "stim_transform": None,
            "resp_transform": None,
            "device": config["device"],
        },
        "patch_dataloader": {
            "batch_size": 4,
            "shuffle": False,
        },
    }
    syn_data_config["patch_dataset"]["data_key"] = dataloaders["mouse_v1"][syn_data_config["data_part"]].data_keys[0]
    print(f"data_part: {syn_data_config['data_part']}   data_key: {syn_data_config['patch_dataset']['data_key']}")

    ### create patches dataloader
    base_dataloader = dataloaders["mouse_v1"][syn_data_config["data_part"]].dataloaders[0]
    patch_dataset = SyntheticDataset(base_dl=base_dataloader, **syn_data_config["patch_dataset"])
    patch_dataloader = DataLoader(patch_dataset, **syn_data_config["patch_dataloader"])
    dl = BatchPatchesDataLoader(dataloader=patch_dataloader)

    ### config
    save_dir = os.path.join(
        DATA_PATH,
        "synthetic_data_mouse_v1_encoder_new_stimuli",
        syn_data_config['patch_dataset']['data_key'],
        syn_data_config["data_part"],
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"{save_dir=}")

    ### save config to parent folder
    with open(os.path.join(os.path.dirname(save_dir), f"config_{syn_data_config['data_part']}.json"), "w") as f:
        json.dump(syn_data_config, f)

    ### save
    sample_idx = 0
    resps_all = []
    for b_idx, (patches, resps, pupil_center) in enumerate(dl):
        if syn_data_config["save_stats"]:
            resps_all.append(resps.cuda())
        for i in range(patches.shape[0]):
            sample_path = os.path.join(save_dir, f"{sample_idx}.pickle")
            with open(sample_path, "wb") as f:
                pickle.dump({
                    "stim": patches[i].cpu(),
                    "resp": resps[i].cpu(),
                    "pupil_center": pupil_center[i].cpu(),
                }, f)
            sample_idx += 1
            if sample_idx >= syn_data_config["max_samples"]:
                break
        if sample_idx >= syn_data_config["max_samples"]:
            break

        if b_idx % 50 == 0:
            print(f"Batch {b_idx} processed")
    print(f"[INFO] Synthetic data generation finished.")

    if syn_data_config["save_stats"]:
        print(f"[INFO] Saving stats to {os.path.join(save_dir, 'stats')}...")
        resps_all = torch.cat(resps_all, dim=0).cpu()
        iqr = torch.quantile(resps_all, 0.75, dim=0) - torch.quantile(resps_all, 0.25, dim=0)
        med = torch.median(resps_all, dim=0).values
        mean = resps_all.mean(dim=0)
        std = resps_all.std(dim=0)
        if not os.path.exists(os.path.join(save_dir, "stats")):
            os.makedirs(os.path.join(save_dir, "stats"))
        else:
            print("[WARNING] stats directory already exists")
        np.save(
            os.path.join(
                save_dir,
                "stats",
                f"responses_iqr.npy"
            ),
            iqr.cpu().numpy(),
        )
        np.save(
            os.path.join(
                save_dir,
                "stats",
                f"responses_mean.npy"
            ),
            mean.cpu().numpy(),
        )
        np.save(
            os.path.join(
                save_dir,
                "stats",
                f"responses_med.npy"
            ),
            med.cpu().numpy(),
        )
        np.save(
            os.path.join(
                save_dir,
                "stats",
                f"responses_std.npy"
            ),
            std.cpu().numpy(),
        )
        print(f"[INFO] Stats saved.")

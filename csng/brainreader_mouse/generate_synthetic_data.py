import os
import matplotlib.pyplot as plt
import numpy as np
import json
import dill
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import lovely_tensors as lt
lt.monkey_patch()

from csng.utils.mix import seed_all
from csng.utils.data import crop, Normalize, PerSampleStoredDataset, MixedBatchLoader
from csng.brainreader_mouse.data import get_brainreader_mouse_dataloaders
from csng.brainreader_mouse.encoder import get_encoder
from csng.mouse_v1.data import get_mouse_v1_dataloaders
from csng.data import get_syn_dataloaders

DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")


### config setup
config = {
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
    },
    "device": os.environ["DEVICE"],
    "seed": 0,

    "encoder_path": os.path.join(DATA_PATH, "models", "encoder_ball.pt"),

}

### base dataloader config
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 1,
    "sessions": list(range(1, 23)),
    "resize_stim_to": None, # keep original size 144x256
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}

### synthetic data config
config["syn_data"] = {
    "data_part_src": "train", # from base_dls
    "data_part_target": "train", # to which folder to save the synthetic data
    "data_key_src": "6", # from base_dls
    "max_samples": 50000, # None or int
    "save_stats": True,
    "patch_dataset": {
        "data_key": "6", # data key on which the encoder was trained
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

config["syn_data"]["path"] = os.path.join(
    DATA_PATH,
    f"synthetic_data_{config['syn_data']['data_key_src']}_{config['syn_data']['data_part_src']}",
    config["syn_data"]["patch_dataset"]["data_key"],
    config["syn_data"]["data_part_target"],
)


class SyntheticDatasetGenerator(Dataset):
    """Extracts patches from given images and encodes them with a pretrained encoder ("on the fly")."""

    def __init__(
        self,
        encoder,
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

        self.encoder = encoder
        self.encoder.eval()

        self.seen_idxs = set()

    def __len__(self):
        return len(self.base_dl)

    def __getitem__(self, idx):
        if idx in self.seen_idxs:
            raise ValueError(f"Index {idx} already seen.")
        self.seen_idxs.add(idx)

        sample = next(self.base_dl_iter)
        img, pupil_center = sample.images.to(self.device), None
        if hasattr(sample, "pupil_center"):
            pupil_center = sample.pupil_center.to(self.device)

        patches, syn_resps = self.extract_patches(img=img, pupil_center=pupil_center)
        if pupil_center is not None:
            return patches, syn_resps, pupil_center.repeat(patches.shape[0], 1)
        return patches, syn_resps

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

        ### merge into a batch
        patches = torch.cat(patches, dim=0).to(self.device)
        assert patches.shape[-2:] == self.patch_shape, f"Expected patch shape {self.patch_shape}, got {patches.shape[-2:]}."

        ### encode patches = get resps
        # if self.expand_stim_for_encoder:
        #     patches_for_encoder = F.interpolate(patches, size=self.patch_size, mode="bilinear", align_corners=False)
        #     ### take only the center of the patch - the encoder's resps cover only the center part
        #     patches = patches[:, :, int(patch_size / 4):int(patch_size / 4) + self.patch_size,
        #                 int(patch_size / 4):int(patch_size / 4) + self.patch_size]
        # patches = self._scale_for_encoder(patches)
        if self.encoder is not None:
            if hasattr(self.encoder, "shifter") and self.encoder.shifter:
                resps = self.encoder(patches, data_key=self.data_key, pupil_center=pupil_center.expand(patches.shape[0], -1))
            else:
                resps = self.encoder(patches, data_key=self.data_key)

        if self.resp_transform is not None:
            resps = self.resp_transform(resps)

        if self.stim_transform is not None:
            patches = self.stim_transform(patches)

        return patches, resps


class BatchPatchesDataLoader():
    """Dataloader that mixes patches from different images within a batch."""

    def __init__(self, dataloader):
        self.dataloader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader_iter)

    def __iter__(self):
        for batch in self.dataloader_iter:
            patches, resps = batch[:2] # (B, N_patches, C, H, W)
            patches = patches.view(-1, *patches.shape[2:])
            resps = resps.view(-1, *resps.shape[2:])

            ### shuffle patch-resp pairs
            idx = torch.randperm(patches.shape[0])
            patches = patches[idx]
            resps = resps[idx].float()

            if len(batch) == 3:
                pupil_center = batch[2].view(-1, *batch[2].shape[2:])[idx]
                yield patches, resps, pupil_center
            yield patches, resps



def generate_synthetic_data(cfg):
    print(f"... Running on {cfg['device']} ...")
    seed_all(cfg["seed"])

    ### load encoder
    encoder = get_encoder(
        ckpt_path=cfg["encoder_path"],
        device=os.environ["DEVICE"],
        eval_mode=True,
    )

    ### init base dataloader
    base_dls = get_brainreader_mouse_dataloaders(cfg["data"]["brainreader_mouse"])["brainreader_mouse"]

    ### synthetic data generator
    print(
        f"data_part_src:\t\t{cfg['syn_data']['data_part_src']}"
        f"\ndata_part_target:\t{cfg['syn_data']['data_part_target']}"
        f"\ndata_key_src:\t\t{cfg['syn_data']['data_key_src']}"
        f"\nmax_samples:\t\t{cfg['syn_data']['max_samples']}"
        f"\nencoder data key:\t{cfg['syn_data']['patch_dataset']['data_key']}"
    )
    ### setup synthetic data dataset and dataloader
    base_dl = base_dls[cfg["syn_data"]["data_part_src"]].dataloaders[
        base_dls[cfg["syn_data"]["data_part_src"]].data_keys.index(cfg["syn_data"]["data_key_src"])
    ]
    patch_dset = SyntheticDatasetGenerator(encoder=encoder, base_dl=base_dl, **cfg["syn_data"]["patch_dataset"])
    patch_dl = DataLoader(patch_dset, **cfg["syn_data"]["patch_dataloader"])
    dl = BatchPatchesDataLoader(dataloader=patch_dl)

    ### prepare synthetic data destination
    print(f"{cfg['syn_data']['path']=}")
    if os.path.exists(cfg["syn_data"]["path"]):
        print("Directory already exists.")
    os.makedirs(cfg["syn_data"]["path"], exist_ok=True)
    # save config and encoder in the parent directory
    with open(os.path.join(os.path.dirname(cfg["syn_data"]["path"]), f"config_{cfg['syn_data']['data_part_target']}.json"), "w") as f:
        json.dump(cfg, f)
    torch.save({
        "encoder": encoder.state_dict(),
        "config": cfg,
    }, os.path.join(os.path.dirname(cfg["syn_data"]["path"]), f"encoder_{cfg['syn_data']['data_part_target']}.pt"), pickle_module=dill)

    ### save synthetic data
    seed_all(cfg["seed"])
    sample_idx = 0
    resps_all = [] # for statistics
    for b_idx, b in enumerate(dl):
        ### b = (patches, resps, pupil_center) or (patches, resps)
        patches, resps = b[:2]

        if cfg["syn_data"]["save_stats"]:
            resps_all.append(resps.cpu())

        ### save each sample separately
        for i in range(patches.shape[0]):
            sample_path = os.path.join(cfg["syn_data"]["path"], f"{sample_idx}.pickle")
            with open(sample_path, "wb") as f:
                to_save = {
                    "stim": patches[i].cpu(),
                    "resp": resps[i].cpu(),
                }
                if len(b) == 3: # add pupil_center if present
                    to_save["pupil_center"] = b[2][i].cpu()
                pickle.dump(to_save, f)
            sample_idx += 1
            if sample_idx >= cfg["syn_data"]["max_samples"]:
                break
        if sample_idx >= cfg["syn_data"]["max_samples"]:
            break

        if b_idx % 50 == 0:
            print(f"Batch {b_idx} processed ({sample_idx} samples saved).")

    ### save statistics of responses
    if cfg["syn_data"]["save_stats"]:
        print(f"[INFO] Saving stats to {os.path.join(cfg['syn_data']['path'], 'stats')}...")
        resps_all = torch.cat(resps_all, dim=0).cpu()
        iqr = torch.quantile(resps_all, 0.75, dim=0) - torch.quantile(resps_all, 0.25, dim=0)
        med = torch.median(resps_all, dim=0).values
        mean = resps_all.mean(dim=0)
        std = resps_all.std(dim=0)
        if not os.path.exists(os.path.join(cfg["syn_data"]["path"], "stats")):
            os.makedirs(os.path.join(cfg["syn_data"]["path"], "stats"))
        else:
            print("[WARNING] stats directory already exists")
        np.save(
            os.path.join(
                cfg["syn_data"]["path"],
                "stats",
                f"responses_iqr.npy"
            ),
            iqr.cpu().numpy(),
        )
        np.save(
            os.path.join(
                cfg["syn_data"]["path"],
                "stats",
                f"responses_mean.npy"
            ),
            mean.cpu().numpy(),
        )
        np.save(
            os.path.join(
                cfg["syn_data"]["path"],
                "stats",
                f"responses_med.npy"
            ),
            med.cpu().numpy(),
        )
        np.save(
            os.path.join(
                cfg["syn_data"]["path"],
                "stats",
                f"responses_std.npy"
            ),
            std.cpu().numpy(),
        )
        print(f"[INFO] Stats saved.")


if __name__ == "__main__":
    generate_synthetic_data(cfg=config)

import os
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from csng.utils.data import NumpyToTensor


class PerSampleStoredDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        latent_dataset_dir,
        resp_transform=None,
        additional_keys=None,
        clamp_neg_resp=False,
        avg_resp=True,
        device="cpu",
    ):
        self.dataset_dir = dataset_dir
        self.latent_dataset_dir = latent_dataset_dir
        self.file_names = np.array(
            [
                f_name
                for f_name in os.listdir(self.dataset_dir)
                if f_name.endswith(".pkl") or f_name.endswith(".pickle")
            ]
        )
        self.parent_dir = Path(self.dataset_dir).parent.absolute()
        self.stim_transform = NumpyToTensor(device=device)
        self.resp_transform = (
            resp_transform
            if resp_transform is not None
            else NumpyToTensor(device=device)
        )
        self.additional_keys = additional_keys
        self.clamp_neg_resp = clamp_neg_resp
        self.avg_resp = avg_resp
        self.keys_to_return = ["images", "responses"]
        self.device = device
        if self.additional_keys is not None:
            self.keys_to_return.extend(self.additional_keys)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]

        with open(os.path.join(self.dataset_dir, f_name), "rb") as f:
            data = pickle.load(f)

        with open(os.path.join(self.latent_dataset_dir, f_name), "rb") as f:
            latent_data = pickle.load(f)

        vals_to_return = [latent_data, data["resp"]]

        if self.stim_transform is not None:
            vals_to_return[0] = self.stim_transform(vals_to_return[0])

        if self.resp_transform is not None:
            vals_to_return[1] = self.resp_transform(vals_to_return[1])

        ### average responses
        if self.avg_resp:
            vals_to_return[1] = vals_to_return[1].mean(axis=0)

        if self.clamp_neg_resp:
            vals_to_return[1].clamp_min_(0)

        ### additional keys
        if self.additional_keys is not None:
            for key in self.additional_keys:
                vals_to_return.append(data[key])

        ### to device
        vals_to_return = [val.to(self.device) for val in vals_to_return]

        ### reverse
        vals_to_return = vals_to_return[::-1]

        return namedtuple("Datapoint", self.keys_to_return)(*vals_to_return)

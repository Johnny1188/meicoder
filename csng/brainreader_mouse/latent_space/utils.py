import os

import numpy as np
import torch
import torchvision


class NumpyToTensor:
    def __init__(self, device="cpu", unsqueeze_dims=None):
        self.unsqueeze_dims = unsqueeze_dims
        self.device = device

    def __call__(self, x, *args, **kwargs):
        if self.unsqueeze_dims is not None:
            x = np.expand_dims(x, self.unsqueeze_dims)
        return torch.from_numpy(x).float().to(self.device)


class Normalize:
    """Class to normalize data."""

    def __init__(
        self, mean, std, center_data=True, clip_min=None, clip_max=None
    ):
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


def get_stim_transform(resize=256, device="cpu"):
    stim_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((36, 64)),
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(
                lambda x: x.to(device, dtype=torch.float16)
            ),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )
    return stim_transform


def get_resp_transform(dataset_dir, device):
    resp_mean = torch.from_numpy(
        np.load(
            os.path.join(dataset_dir, str(1), "stats", "responses_mean.npy")
        )
    ).to(device)
    resp_std = torch.from_numpy(
        np.load(
            os.path.join(dataset_dir, str(1), "stats", "responses_std.npy")
        )
    ).to(device)
    resp_transform = torchvision.transforms.Compose(
        [
            NumpyToTensor(device=device),
            Normalize(mean=resp_mean, std=resp_std),
        ]
    )
    return resp_transform

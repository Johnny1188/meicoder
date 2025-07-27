import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import numpy as np
from collections import namedtuple, defaultdict


from csng.utils.data import MixedBatchLoader, NumpyToTensor, Normalize, PerSampleStoredDataset


class DictWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        stim, resp = self.dataset[idx]
        Datapoint = namedtuple('Datapoint', ['responses', 'images'])
        return Datapoint(responses=resp, images=stim)


def get_allen_dataloaders(config):
    DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cae", "data")
    SESS_ID = "allen"
    dls = defaultdict(dict)

    # ### prepare stim transforms
    # stim_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToPILImage(),
    # ])
    # # by default resize to 36x64
    # if config.get("resize_stim_to", (36, 64)) is not None:
    #     stim_transform.transforms.append(
    #         torchvision.transforms.Resize(config.get("resize_stim_to", (36, 64)))
    #     )
    # stim_transform.transforms.extend([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Lambda(lambda x: x.to(config["device"])),
    # ])
    # # normalize stimuli
    # if config["normalize_stim"]:
    #     stim_mean = np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "stimuli_mean.npy")).item()
    #     stim_std = np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "stimuli_std.npy")).item()
    #     stim_transform.transforms.append(
    #         torchvision.transforms.Normalize(mean=stim_mean, std=stim_std)
    #     )

    # ### prepare resp transforms
    # resp_transform = torchvision.transforms.Compose([
    #     NumpyToTensor(device=config["device"]),
    # ])
    # if config["normalize_resp"]:
    #     resp_mean = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_mean.npy"))).to(config["device"])
    #     resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_std.npy"))).to(config["device"])
    #     resp_transform.transforms.append(
    #         Normalize(mean=resp_mean, std=resp_std)
    #     )
    # elif config["div_resp_by_std"]:
    #     resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, str(sess_id), "stats", "responses_std.npy"))).to(config["device"])
    #     resp_transform.transforms.append(
    #         Normalize(mean=0, std=resp_std)
    #     )


    ### original data preparation
    # --- Load and Prepare Data ---
    print("Loading data...")
    # These files need to be present in the './data' directory
    try:
        img_train_np = np.load(os.path.join(DATA_PATH, 'movie_03_train_pic_1999_VISp_800.npy'))
        img_test_np = np.load(os.path.join(DATA_PATH, 'movie_03_test_pic_1999_VISp_800.npy'))
        spike_train_np = np.load(os.path.join(DATA_PATH, 'movie_03_train_spike_1999_VISp_800.npy'))
        spike_test_np = np.load(os.path.join(DATA_PATH, 'movie_03_test_spike_1999_VISp_800.npy'))
        # img_train_np = np.random.rand(100, 1, 256, 256)  # Dummy data for testing
        # img_test_np = np.random.rand(20, 1, 256, 256)  # Dummy data for testing
        # spike_train_np = np.random.rand(100, 800)  # Dummy data for testing
        # spike_test_np = np.random.rand(20, 800)  # Dummy data for testing
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Make sure the .npy files are in the '{DATA_PATH}' directory.")
        print(e)
        exit()

    # Add a channel dimension to images for Conv2D: (N, H, W) -> (N, 1, H, W)
    if img_train_np.ndim == 3:
        img_train_np = np.expand_dims(img_train_np, axis=1)
    if img_test_np.ndim == 3:
        img_test_np = np.expand_dims(img_test_np, axis=1)

    # Convert numpy arrays to PyTorch tensors
    spike_train = torch.from_numpy(spike_train_np).float()
    img_train = torch.from_numpy(img_train_np).float()
    spike_test = torch.from_numpy(spike_test_np).float()
    img_test = torch.from_numpy(img_test_np).float()

    # Z-score images if specified
    if config.get("zscore_images", False):
        img_train = (img_train - img_train.mean(dim=(0, 2, 3), keepdim=True)) / img_train.std(dim=(0, 2, 3), keepdim=True)
        img_test = (img_test - img_train.mean(dim=(0, 2, 3), keepdim=True)) / img_train.std(dim=(0, 2, 3), keepdim=True)

    # Divide responses by standard deviation if specified
    if config.get("div_resp_by_std", False):
        resp_std = spike_train.std(dim=0, keepdim=True)
        resp_std[resp_std < 0.01 * resp_std.mean()] = 0.01 * resp_std.mean()
        spike_train = spike_train / resp_std
        spike_test = spike_test / resp_std

    # Clamp negative responses if specified
    if config.get("clamp_neg_resp", False):
        spike_train = torch.clamp(spike_train, min=0)
        spike_test = torch.clamp(spike_test, min=0)

    # Create TensorDatasets and DataLoaders
    test_dataset = TensorDataset(img_test, spike_test)
    train_dataset = TensorDataset(img_train, spike_train)
    train_size = int((1 - config["val_split_frac"]) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    rand_gen = torch.Generator().manual_seed(config["val_split_seed"]) if "val_split_seed" in config else None
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=rand_gen)

    # Wrap datasets in DictWrapperDataset for consistency
    train_dataset = DictWrapperDataset(train_dataset)
    val_dataset = DictWrapperDataset(val_dataset)
    test_dataset = DictWrapperDataset(test_dataset)

    # Create DataLoaders
    dls_out = {"allen": {
        "train": DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=config.get("drop_last", False)),
        "val": DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=config.get("drop_last", False)),
        "test": DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=config.get("drop_last", False)),
    }}
    print("Data loaded and prepared.")

    return dls_out

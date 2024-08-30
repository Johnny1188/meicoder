import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import re
from functools import wraps
import time


def get_corr(x, y):
    return torch.corrcoef(torch.stack((x.flatten(), y.flatten()), dim=0))[0, 1]


def crop(x, win):
    ### win: (x1, x2, y1, y2) or (slice(x1, x2), slice(y1, y2) or (height, width) of middle patch
    if win == None:
        return x
    if isinstance(win[0], int) and len(win) == 4: # (x1, x2, y1, y2)
        if x.shape[-2] == win[1] - win[0] and x.shape[-1] == win[3] - win[2]:
            return x
        return x[..., win[0]:win[1], win[2]:win[3]]
    elif isinstance(win[0], int) and len(win) == 2: # (height, width) of middle patch
        if x.shape[-2] == win[0] and x.shape[-1] == win[1]:
            return x
        return x[..., (x.shape[-2] - win[0])//2:(x.shape[-2] + win[0])//2, (x.shape[-1] - win[1])//2:(x.shape[-1] + win[1])//2]
    else: # (slice(x1, x2), slice(y1, y2))
        if x.shape[-2] == win[0].stop - win[0].start and x.shape[-1] == win[1].stop - win[1].start:
            return x
        return x[..., win[0], win[1]]


def standardize(x, dim=(1,2,3), eps=1e-8, min=None, max=None, inplace=False):
    ### to [0,1] range
    x_min = x.amin(dim=dim, keepdim=True) if min is None else min
    x_max = x.amax(dim=dim, keepdim=True) if max is None else max

    if inplace:
        x -= x_min
        x /= (x_max - x_min + eps)
        return x
    else:
        return (x - x_min) / (x_max - x_min + eps)


def normalize(x, dim=(1,2,3), mean=None, std=None, inplace=False):
    ### mean 0, std 1
    x_mean = x.mean(dim=dim, keepdim=True) if mean is None else mean
    x_std = x.std(dim=dim, keepdim=True) if std is None else std

    if inplace:
        x -= x_mean
        x /= (x_std + 1e-8)
        return x
    else:
        return (x - x_mean) / (x_std + 1e-8)


def get_mean_and_std(dataset=None, dataloader=None, verbose=False):
    """ Compute the mean and std value of dataset. """
    assert dataset is not None or dataloader is not None, "Either dataset or dataloader must be provided."
    if dataloader is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        data_len = len(dataset)
    else:
        data_len = len(dataloader.dataset)

    mean_inputs, std_inputs = torch.zeros(3), torch.zeros(3)
    mean_targets, std_targets = torch.zeros(1), torch.zeros(1)
    
    print('==> Computing mean and std..')
    # for inp_idx, (inputs, targets, _, _) in enumerate(dataloader):
    for inp_idx, (inputs, targets) in enumerate(dataloader):
        for c in range(inputs.size(1)):
            mean_inputs[c] += inputs[:,c,:,:].cpu().mean()
            std_inputs[c] += inputs[:,c,:,:].cpu().std()
        mean_targets += targets.cpu().mean()
        std_targets += targets.cpu().std()
        
        if verbose and inp_idx % 1000 == 0:
            print(f"Processed {inp_idx} / {data_len} samples.")
    mean_inputs.div_(len(dataset))
    std_inputs.div_(len(dataset))
    mean_targets.div_(len(dataset))
    std_targets.div_(len(dataset))
    
    return {
        "inputs": {
            "mean": mean_inputs,
            "std": std_inputs
        },
        "targets": {
            "mean": mean_targets,
            "std": std_targets
        }
    }


def plot_comparison(target, pred, target_title="Target", pred_title="Reconstructed", n_cols=8, show=True, save_to=None):
    n_imgs = (target.shape[0], pred.shape[0])
    n_rows_per_group = (1 + (n_imgs[0]-1)//n_cols, 1 + (n_imgs[1]-1)//n_cols)

    ### plot comparison
    h_mul_factor = 3 * (target.shape[-2] / target.shape[-1])
    # w_mul_factor = 0.4 + (target.shape[-1] / target.shape[-2])
    fig = plt.figure(figsize=(22, 2.5 + h_mul_factor * sum(n_rows_per_group)))
    # fig = plt.figure(figsize=(n_cols * w_mul_factor, 3 + sum(n_rows_per_group) * h_mul_factor))
    
    for i in range(max(n_imgs)):
        ### target
        if i < n_imgs[0]:
            ax = fig.add_subplot(sum(n_rows_per_group), n_cols, i + 1)
            ax.imshow(target[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(target_title, fontsize=16, fontweight="bold", loc="left")
        
        ### reconstructed
        if i < n_imgs[1]:
            ax = fig.add_subplot(sum(n_rows_per_group), n_cols, i + 1 + n_cols * n_rows_per_group[0])
            ax.imshow(pred[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(pred_title, fontsize=16, fontweight="bold", loc="left")
    if show:
        plt.show()

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight", pad_inches=0.1)
        # print(f"Saved to {save_to}.")

    plt.close(fig)

    return fig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningStats:
    """Class to calculate running standard deviation for random vectors."""

    def __init__(self, num_components, lib="numpy", device="cpu"):
        """
        Initialize the RunningStdDeviation instance.

        Args:
            num_components (int): Number of components in each random vector.
        """
        self.num_components = num_components
        self.lib = lib
        self.device = device
        self.count = 0

        if lib == "numpy":
            self.mean = np.zeros(num_components)
            self.M2 = np.zeros(num_components)
        elif lib == "torch":
            self.mean = torch.zeros(num_components).to(device)
            self.M2 = torch.zeros(num_components).to(device)

    def update(self, values):
        """
        Update the running standard deviation with a new random vector.

        Args:
            values (numpy.ndarray): Array representing the random vector.

        Raises:
            ValueError: If the length of the input vector does not match the number of components.
        """
        if values.shape[-1] != self.num_components:
            raise ValueError("Number of components does not match")

        if len(values.shape) == 1: # batch
            values = values.reshape(1, -1)

        self.count += values.shape[0]
        delta = (values - self.mean).sum(0)
        self.mean += delta / self.count
        delta2 = (values - self.mean).sum(0)
        self.M2 += delta * delta2

    def get_std(self):
        """
        Calculate the running standard deviation for each component random variable.

        Returns:
            numpy.ndarray: Array containing the standard deviation for each component.
        """
        if self.count < 2:
            return np.zeros(self.num_components) if self.lib == "numpy" else torch.zeros(self.num_components).to(self.device)
        return \
            np.sqrt(np.clip(self.M2 / (self.count - 1), 1e-6, None)) if self.lib == "numpy" else \
            torch.sqrt(torch.clip(self.M2 / (self.count - 1), 1e-6, None))

    def get_mean(self):
        """
        Calculate the running mean for each component random variable.

        Returns:
            numpy.ndarray: Array containing the mean for each component.
        """
        return self.mean


class Normalize:
    """ Class to normalize data. """

    def __init__(self, mean, std, center_data=True, clip_min=None, clip_max=None):
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


def plot_losses(history, show=True, save_to=None, epoch=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    if show:
        plt.show()
    
    if save_to:
        fig.savefig(save_to)

    plt.close(fig)


def build_layers(
        in_channels,
        layers_config,
        act_fn=nn.ReLU,
        out_act_fn=nn.Identity,
        dropout=0.0,
        batch_norm=False,
        layer_norm=False,
    ):
        assert not (batch_norm and layer_norm), "Only one of batch_norm and layer_norm can be True."
        ### build layers from tuple configs
        layers = []
        for l_i, layer_config in enumerate(layers_config):
            if layer_config[0] == "fc":
                layer_type, out_channels = layer_config
                layers.append(nn.Linear(in_channels, out_channels))
            elif layer_config[0] == "unflatten":
                layer_type, in_dim, unflattened_size = layer_config
                layers.append(nn.Unflatten(in_dim, unflattened_size))
                out_channels = unflattened_size[0]
            elif layer_config[0] == "flatten":
                layer_type, start_dim, end_dim, out_channels = layer_config
                layers.append(nn.Flatten(start_dim, end_dim))
            elif layer_config[0] == "maxpool":
                layer_type, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                )
                out_channels = in_channels
            elif layer_config[0] == "upsample":
                layer_type, scale_factor = layer_config
                layers.append(nn.Upsample(scale_factor=scale_factor))
                out_channels = in_channels
            elif layer_config[0] == "deconv":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            elif layer_config[0] == "conv":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            elif layer_config[0] == "conv1d":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            else:
                raise ValueError(f"layer_type {layer_config[0]} not recognized")

            if l_i < len(layers_config) - 1 and layer_type in ["fc", "conv", "deconv", "conv1d"]:
                ### add batch norm, activation, dropout
                if batch_norm:
                    if layer_type in ["fc", "conv1d"]:
                        layers.append(nn.BatchNorm1d(out_channels))
                    else:
                        layers.append(nn.BatchNorm2d(out_channels))
                elif layer_norm:
                    if layer_type in ["fc", "conv1d"]:
                        layers.append(nn.LayerNorm(out_channels))
                    else:
                        layers.append(nn.GroupNorm(1, out_channels))
                layers.append(act_fn())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            elif l_i == len(layers_config) - 1:
                ### add output activation
                layers.append(out_act_fn())

            in_channels = out_channels

        return nn.Sequential(*layers)


def dict_to_str(d, as_filename=False):
    def print_val(v):
        if type(v) == dict:
            return dict_to_str(v)
        elif type(v) in (list, tuple):
            return [print_val(_v) for _v in v]
        elif type(v) == str or v == None:
            return v
        else:
            return f"{v:.3f}"
    if as_filename:
        return "___".join([f"{k}={print_val(v)}" for k, v in d.items()])
    else:
        return ", ".join([f"{k}: {print_val(v)}" for k, v in d.items()])


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def seed_all(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # multi-GPU.

        ### deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_config_paths(config, new_data_path):
    def update_path_str(old_path, new_path):
        ### count number of folders in new_dat_path from the root "/1/2/3/4/5" -> 5,
        ###   and then replace only the first 5 folders in the old path
        old_data_path_split = old_path.split("/")
        new_data_path_split = new_path.split("/")
        n_folders = len(new_data_path_split)
        return os.path.join("/", *new_data_path_split[:n_folders], *old_data_path_split[n_folders:])
        # update_path = "/".join(new_data_path_split[:n_folders])
        # if len(old_data_path_split[n_folders:]) > 0:
        #     update_path += "/"
        # return "/".join(new_data_path_split[:n_folders]) \
        #     + "/".join(old_data_path_split[n_folders:])

    ### update paths that use remote DATA_PATH with new_data_path
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict):
                update_config_paths(v, new_data_path)
            elif isinstance(v, list) or isinstance(v, tuple):
                for _v in v:
                    update_config_paths(_v, new_data_path)
            elif k in ["data_dir", "ckpt_path", "save_path"] and isinstance(v, str):
                config[k] = update_path_str(v, new_data_path)
            else:
                update_config_paths(v, new_data_path)
    elif isinstance(config, list) or isinstance(config, tuple):
        for v in config:
            update_config_paths(v, new_data_path)

    return config


def correct_path(path_to_correct, new_data_path_start):
    # TODO: not tested
    ### traverse down to the root from the current location until the location doesn't exist, then replace the rest of the path
    path_to_correct_split = path_to_correct.split("/")
    from pathlib import Path
    curr_path = Path(os.getcwd())
    i = len(path_to_correct_split) - 2
    while i >= 0:
        dir_expected = path_to_correct_split[i]
        dir_true = curr_path.parent.absolute().name

        if dir_expected != dir_true:
            break

        i -= 1
        curr_path = curr_path.parent

    return os.path.join(new_data_path_start, *path_to_correct_split[i+1:])


def timeit(func):
    """
    Source: https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"> {func.__name__}(): {total_time / 60:.2f} mins = {total_time:.2f} secs = {total_time * 1000:.2f} ms = {total_time * 1000 * 1000:.2f} µs")
        return result
    return timeit_wrapper

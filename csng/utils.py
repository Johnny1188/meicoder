import torch
import numpy as np
import matplotlib.pyplot as plt


def get_corr(x, y):
    return torch.corrcoef(torch.stack((x.flatten(), y.flatten()), dim=0))[0, 1]


def crop(x, win):
    ### win: (x1, x2, y1, y2) or (slice(x1, x2), slice(y1, y2)
    return x[..., win[0]:win[1], win[2]:win[3]] \
        if isinstance(win[0], int) else x[..., win[0], win[1]]


def standardize(x, dim=(1,2,3), eps=1e-8, inplace=False):
    ### [0,1] range
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)

    if inplace:
        x -= x_min
        x /= (x_max - x_min + eps)
        return x
    else:
        return (x - x_min) / (x_max - x_min + eps)


def normalize(x, dim=(1,2,3), inplace=False):
    ### mean 0, std 1
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)

    if inplace:
        x -= x_mean
        x /= x_std
        return x
    else:
        return (x - x_mean) / x_std

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
    for inp_idx, (inputs, targets) in enumerate(dataloader):
        for c in range(inputs.size(1)):
            mean_inputs[c] += inputs[:,c,:,:].mean()
            std_inputs[c] += inputs[:,c,:,:].std()
        mean_targets += targets.mean()
        std_targets += targets.std()
        
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

def plot_comparison(target, pred, target_title="Target", pred_title="Reconstructed", n_cols=8, save_to=None):
    n_imgs = (target.shape[0], pred.shape[0])
    n_rows_per_group = (1 + (n_imgs[0]-1)//n_cols, 1 + (n_imgs[1]-1)//n_cols)

    ### plot comparison
    fig = plt.figure(figsize=(22, 3 + 1.5 * sum(n_rows_per_group)))
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
    plt.show()

    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {save_to}.")


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

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, values):
        return (values - self.mean) / self.std

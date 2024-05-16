import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import dill

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


def plot_meis():
    ### config
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "meis_path": os.path.join(DATA_PATH, "meis", "21067-10-18", "meis.pt"),
        "plots_dir": os.path.join(DATA_PATH, "meis", "21067-10-18", "plots"),
        "plot_n_neurons": 3, # number of cells to process at once
        "randomly_choose": True,
        "crop_win": None,
    }

    ### create dir
    os.makedirs(config["plots_dir"], exist_ok=True)

    ### load
    meis = torch.load(config["meis_path"], pickle_module=dill)["meis"] # (n_neurons, C, H, W)
    
    ### crop
    if config["crop_win"] is not None:
        meis = crop(meis, config["crop_win"])
    
    ### select to plot
    if config["randomly_choose"]:
        neurons_idxs = np.random.choice(meis.shape[0], size=config["plot_n_neurons"], replace=False)
    else:
        neurons_idxs = np.arange(config["plot_n_neurons"])
    meis_to_plot = meis[neurons_idxs]

    ### plot
    fig = plt.figure(figsize=(10, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, config["plot_n_neurons"]), direction="row", axes_pad=0.03, share_all=True)
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    def plot_img(img, curr_ax_idx):
        grid[curr_ax_idx].imshow(img.permute(1,2,0), "gray")
        for d in ("top", "right", "left", "bottom"):
            grid[curr_ax_idx].spines[d].set_visible(False)

    def set_title(ax, title):
        ax.set_title(title, fontsize=10, rotation=0, va="baseline")

    for ax_idx, (neuron_idx, mei) in enumerate(zip(neurons_idxs, meis_to_plot)):
        set_title(ax=grid[ax_idx], title=f"Neuron #{neuron_idx}")
        plot_img(mei, curr_ax_idx=ax_idx)

    plt.show()

    file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for f_type in ("png", "pdf"):
        save_path = os.path.join(config["plots_dir"], f"{file_name}.{f_type}")
        fig.savefig(save_path, bbox_inches="tight")

    print(f"[INFO] MEIs plotted to {config['plots_dir']} (file {file_name}).")


if __name__ == "__main__":
    plot_meis()

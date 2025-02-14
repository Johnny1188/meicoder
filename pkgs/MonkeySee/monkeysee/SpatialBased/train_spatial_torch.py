import os
import json
import wandb
from datetime import datetime
import dill
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()

from monkeysee.SpatialBased.discriminator import Discriminator
from monkeysee.SpatialBased.generator import Generator
from csng.data import get_dataloaders
from csng.utils.mix import seed_all

DATA_PATH_BRAINREADER = os.path.join(os.environ["DATA_PATH"], "brainreader")
DATA_PATH_MONKEYSEE = os.path.join(os.environ["DATA_PATH"], "monkeysee")


### load RFs
def get_RFs(meis_path=None, spatial_embeddings_path=None, device="cpu"):
    assert (meis_path is None) != (spatial_embeddings_path is None), \
        "Exactly one of `meis_path` and `spatial_embeddings_path` must be provided."

    ### load
    if meis_path:
        RFlocs = torch.load(meis_path, pickle_module=dill)["meis"].to(device)  # (n_neurons, 1, height, width)
    elif spatial_embeddings_path:
        RFlocs = torch.load(spatial_embeddings_path, pickle_module=dill)["embedding"]["W"].detach().unsqueeze(1).to(device)  # (n_neurons, 1, height, width)
        RFlocs.requires_grad_(False)

    ### min-max normalize
    dims = list(range(RFlocs.dim()))[1:]
    RFlocs = (RFlocs - RFlocs.amin(dim=dims, keepdim=True)) / (RFlocs.amax(dim=dims, keepdim=True) - RFlocs.amin(dim=dims, keepdim=True))

    return RFlocs.permute(1, 0, 2, 3)  # (1, n_neurons, height, width)

### contextualize RFs using brain signal
def get_inputsROI(brain, RFs, n_groups=None, sum_out=True):
    assert (n_groups is not None) == sum_out, "n_groups must be provided if and only if sum_out is True."

    if brain.ndim == 3:
        brain = brain.unsqueeze(-1)

    channels = RFs * brain  # (B, n_neurons, height, width)

    if sum_out:
        if n_groups is not None:
            n_neurons_per_group = [channels.size(1) // n_groups] * n_groups
            if channels.size(1) % n_groups != 0:
                n_neurons_per_group[-1] += channels.size(1) % n_groups  # add the remainder to the last group

            ### sum within groups
            channels = torch.cat([
                channels[:, sum(n_neurons_per_group[:i]):sum(n_neurons_per_group[:i+1])].sum(dim=1, keepdim=True)
                for i in range(n_groups)
            ], dim=1)  # (B, n_groups, height, width)
        else:
            channels = channels.sum(dim=1, keepdim=True)  # (B, 1, height, width)
    
    return channels

def get_inputs(brains, config, transform_inputs_fn, RFs=None):
    if config["decoder"]["gen"]["inverse_retinotopic_mapping_cfg"] is None:
        inputs = get_inputsROI(
            brain=brains.unsqueeze(-1),
            RFs=RFs,
            n_groups=config["decoder"]["gen"]["input_channels"],
            sum_out=config["decoder"]["sum_rfs_out"],
        )
    else:
        inputs = brains.squeeze(-1)

    ### transform inputs
    if transform_inputs_fn is not None:
        inputs = transform_inputs_fn(inputs)

    return inputs

### utils
def compute_mean_std(dl, config, RFs):
    mean, channels_sqrd_sum = 0, 0
    num_batches = len(dl)
    for batch in tqdm(dl, total=len(dl)):
        brains = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1)
        inputs = get_inputs(brains=brains, config=config, transform_inputs_fn=None, RFs=RFs)
        mean += torch.mean(inputs, dim=[0, 2, 3]).to(torch.float64) / num_batches
        channels_sqrd_sum += torch.mean(inputs ** 2, dim=[0, 2, 3]).to(torch.float64) / num_batches

    std = (channels_sqrd_sum - mean ** 2) ** 0.5
    return mean.to(torch.float32), std.to(torch.float32)



### setup config
cfg = {
    "device": os.environ.get("DEVICE", "cpu"),
    "seed": 0,
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "monkeysee",
    },
}
cfg["data"]["brainreader_mouse"] = {
    "device": cfg["device"],
    "mixing_strategy": cfg["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 32,
    # "sessions": list(range(1, 23)),
    "sessions": [6],
    "resize_stim_to": (36, 64),
    "normalize_stim": True,
    "normalize_resp": True,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}
cfg["decoder"] = {
    "gen": {
        "input_channels": 480,
        "normalized": cfg["data"]["brainreader_mouse"]["normalize_stim"],
        "inverse_retinotopic_mapping_cfg": None,

        # "input_channels": 1,
        # "inverse_retinotopic_mapping_cfg": {
        #     "n_neurons": 1,
        #     "height": 36,
        #     "width": 64,
        #     "sum_maps": True,
        #     "device": cfg["device"],
        # },

        "alpha": 0.01,
        "beta_vgg": 0.9,
        "beta_pix": 0.09,
        "lr": 0.0002,
        # "lr": 0.001,
        "betas": (0.5, 0.999),
        # "betas": (0.9, 0.999),
        "weight_decay": 0,
        # "weight_decay": 3e-4,
    },
    "dis": {
        "input_channels": 1,
        "lr": 0.0002,
        # "lr": 0.001,
        "betas": (0.5, 0.999),
        "weight_decay": 0,
        # "weight_decay": 3e-4,
    },
    "sum_rfs_out": True,
    "standardize_inputs": True,
    "epochs": 300,
    "load_ckpt": None,
    # "load_ckpt": "/home/jan/Desktop/Dev/MonkeySee/data/models/03-02-2025_00-37"
}
cfg["rfs"] = {
    "meis_path": None,
    # "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis", "6", "meis.pt"),
    # "spatial_embeddings_path": None,
    "spatial_embeddings_path": os.path.join(DATA_PATH_MONKEYSEE, "spatial_embedding", "08-02-2025_13-33", "embedding.pt"),
    "device": cfg["device"],
}


if __name__ == '__main__':
    print(f"... Running on {cfg['device']} ...")

    assert cfg["data"]["brainreader_mouse"]["sessions"] == [6], \
        "Only session 6 is supported for now."

    ### config
    cfg["run_name"] = datetime.now().strftime("%d-%m-%Y_%H-%M")
    cfg["save_dir"] = os.path.join(os.environ["DATA_PATH"], "monkeysee", "runs", cfg["run_name"])
    os.makedirs(cfg["save_dir"], exist_ok=True)
    print(f"[INFO] Saving to {cfg['save_dir']}")
    print(f"[INFO] Run name: {cfg['run_name']}")

    ### prepare data
    seed_all(cfg["seed"])
    RFs = get_RFs(**cfg["rfs"])

    ### w&b
    wdb_run = wandb.init(**cfg["wandb"], name=cfg["run_name"], config=cfg, save_code=True,
        tags=["MonkeySee", "Inv-Ret-Map" if cfg["decoder"]["gen"]["inverse_retinotopic_mapping_cfg"] is not None else "MEIs"])
    wdb_run.log_code(".", include_fn=lambda path, root: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") or path.endswith(".yml"))

    ### initialize models
    seed_all(cfg["seed"])
    discriminator = Discriminator(**cfg["decoder"]["dis"], device=cfg["device"])
    generator = Generator(**cfg["decoder"]["gen"], device=cfg["device"])
    wdb_run.watch(discriminator, log="all")
    wdb_run.watch(generator, log="all")
    print(f"[INFO] Discriminator: {discriminator}\n\n[INFO] Generator: {generator}")

    ### load checkpoints
    if cfg["decoder"]["load_ckpt"] is not None:
        print(f"[INFO] Loading checkpoint from {cfg['decoder']['load_ckpt']} ...")
        ckpt_dis = torch.load(f"{cfg['decoder']['load_ckpt']}/discriminator.pt", pickle_module=dill)
        ckpt_gen = torch.load(f"{cfg['decoder']['load_ckpt']}/generator.pt", pickle_module=dill)
        discriminator.load_state_dict(ckpt_dis["state_dict"])
        generator.load_state_dict(ckpt_gen["state_dict"])
        history = ckpt_gen["history"]
    else:
        print("[INFO] No checkpoint loaded.")
        history = {k: [] for k in ['D_loss_train', 'G_loss_train', 'G_loss_D_train', 'G_loss_vgg_train', 'G_loss_pix_train', 'G_loss_vgg_val', 'G_loss_pix_val']}

    ### collect statistics
    transform_inputs = lambda x: x
    if cfg["decoder"]["standardize_inputs"]:
        print("[INFO] Collecting statistics ...")
        dls, _ = get_dataloaders(config=cfg)
        train_dl = dls["train"]["brainreader_mouse"]
        mean, std = compute_mean_std(dl=train_dl, config=cfg, RFs=RFs)
        print(f"  mean: {mean}\n  std: {std}")
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1)
        transform_inputs = lambda x: (x - mean) / (std + 1e-6)

    ### train
    best_l1 = {"epoch": None, "val_loss_vgg": np.inf, "val_loss_l1": np.inf, "recon": None, "generator": None, "discriminator": None}
    best_vgg = {"epoch": None, "val_loss_vgg": np.inf, "val_loss_l1": np.inf, "recon": None, "generator": None, "discriminator": None}
    seed_all(cfg["seed"])
    for ep in range(cfg["decoder"]["epochs"]):
        generator.train()
        discriminator.train()
        dls, _ = get_dataloaders(config=cfg)
        train_dl, val_dl = dls["train"]["brainreader_mouse"], dls["val"]["brainreader_mouse"]
        print(
            f"[E {ep+1}/{cfg['decoder']['epochs']}  {datetime.now().strftime('%H:%M:%S')}]\n  "
            + '\n  '.join([f'{k}: {np.mean(v[-len(train_dl):]):.4f}' for k, v in history.items()] if ep > 0 else [])
        )

        for batch in tqdm(train_dl, total=len(train_dl)):
            ### prepare inputs
            brains = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
            targets = torch.cat([dp["stim"] for dp in batch], dim=0).to(cfg["device"])
            inputs = get_inputs(brains=brains, config=cfg, transform_inputs_fn=transform_inputs, RFs=RFs)

            ### compute losses
            dis_loss = discriminator.train_model(
                g=generator,
                x=inputs,
                y=targets,
                step=True,
            )
            gen_loss_total, gen_loss_dis, gen_loss_vgg, gen_loss_pix = generator.train_model(
                d=discriminator,
                x=inputs,
                y=targets,
                step=True,
            )

            ### save losses
            history["D_loss_train"].append(dis_loss)
            history["G_loss_train"].append(gen_loss_total)
            history["G_loss_D_train"].append(gen_loss_dis)
            history["G_loss_vgg_train"].append(gen_loss_vgg)
            history["G_loss_pix_train"].append(gen_loss_pix)

        ### eval generator
        generator.eval()
        with torch.no_grad():
            vgg_loss, l1_loss, n_samples = 0, 0, 0
            for batch in val_dl:
                brains = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
                targets = torch.cat([dp["stim"] for dp in batch], dim=0).to(cfg["device"])
                inputs = get_inputs(brains=brains, config=cfg, transform_inputs_fn=transform_inputs, RFs=RFs)

                recons, inv_ret_maps = generator(inputs, return_inv_ret_maps=True)

                _vgg_loss = generator._lossfun._vgg(recons, targets, reduction="none")
                vgg_loss += torch.stack([vgg_layer_loss.mean(dim=(1, 2, 3)) for vgg_layer_loss in _vgg_loss], dim=1).mean(dim=1).sum().item()
                l1_loss += F.l1_loss(recons, targets, reduction="none").mean(dim=(1, 2, 3)).sum().item()
                n_samples += len(recons)
            vgg_loss /= n_samples
            l1_loss /= n_samples
            print(f"  val. VGG: {vgg_loss:.4f}\n  val. L1: {l1_loss:.4f}")
            history["G_loss_vgg_val"].append(vgg_loss)
            history["G_loss_pix_val"].append(l1_loss)

            ### save if best
            if l1_loss < best_l1["val_loss_l1"]:
                best_l1["epoch"] = ep
                best_l1["val_loss_vgg"] = vgg_loss
                best_l1["val_loss_l1"] = l1_loss
                best_l1["generator"] = deepcopy(generator.state_dict())
                best_l1["discriminator"] = deepcopy(discriminator.state_dict())

            if vgg_loss < best_vgg["val_loss_vgg"]:
                best_vgg["epoch"] = ep
                best_vgg["val_loss_vgg"] = vgg_loss
                best_vgg["val_loss_l1"] = l1_loss
                best_vgg["generator"] = deepcopy(generator.state_dict())
                best_vgg["discriminator"] = deepcopy(discriminator.state_dict())

        ### plot reconstructions
        fig = plt.figure(figsize=(15, 6))
        for i in range(min(5, len(recons))):
            ax = fig.add_subplot(3, min(5, len(recons)), i+1)
            ax.imshow(targets[i].cpu().numpy().squeeze(), cmap='gray')
            ax.axis('off')
            ax = fig.add_subplot(3, min(5, len(recons)), i+min(5, len(recons))+1)
            ax.imshow(recons[i].cpu().numpy().squeeze(), cmap='gray')
            ax.axis('off')
            if cfg["decoder"]["gen"]["inverse_retinotopic_mapping_cfg"] is not None:
                if inv_ret_maps.size(1) == 1:
                    ax = fig.add_subplot(3, min(5, len(recons)), i+2*min(5, len(recons))+1)
                    ax.imshow(inv_ret_maps[i].cpu().numpy().squeeze(), cmap='gray')
                    ax.axis('off')
                else:
                    ax = fig.add_subplot(3, min(5, len(recons)), i+2*min(5, len(recons))+1)
                    ax.imshow(inv_ret_maps[i][i].unsqueeze(0).cpu().numpy().squeeze(), cmap='gray')
                    ax.axis('off')
        fig.tight_layout()
        fig.savefig(f"{cfg['save_dir']}/recons_{ep}.png")
        if best_l1["epoch"] == ep:
            fig.savefig(f"{cfg['save_dir']}/recons_best_l1.png")
            best_l1["recon"] = fig
        if best_vgg["epoch"] == ep:
            fig.savefig(f"{cfg['save_dir']}/recons_best_vgg.png")
            best_vgg["recon"] = fig
        plt.close(fig)

        ### log
        wdb_run.log({
            "D_loss_train": np.mean(history["D_loss_train"][-len(train_dl):]),
            "G_loss_train": np.mean(history["G_loss_train"][-len(train_dl):]),
            "G_loss_D_train": np.mean(history["G_loss_D_train"][-len(train_dl):]),
            "G_loss_vgg_train": np.mean(history["G_loss_vgg_train"][-len(train_dl):]),
            "G_loss_vgg_val": history["G_loss_vgg_val"][-1],
            "G_loss_pix_val": history["G_loss_pix_val"][-1],
            "reconstructions": fig,
            "best_l1_reconstructions": best_l1["recon"],
            "best_vgg_reconstructions": best_vgg["recon"],
        })
        plt.close(fig)

        ### save
        torch.save({
            "state_dict": generator.state_dict(),
            "best_l1": best_l1["generator"],
            "best_l1_val_loss_l1": best_l1["val_loss_l1"],
            "best_l1_val_loss_vgg": best_l1["val_loss_vgg"],
            "best_vgg": best_vgg["generator"],
            "best_vgg_val_loss_l1": best_vgg["val_loss_l1"],
            "best_vgg_val_loss_vgg": best_vgg["val_loss_vgg"],
            "config": cfg,
            "history": history,
        }, f"{cfg['save_dir']}/generator.pt", pickle_module=dill)
        torch.save({
            "state_dict": discriminator.state_dict(),
            "best_l1": best_l1["discriminator"],
            "best_l1_val_loss_l1": best_l1["val_loss_l1"],
            "best_l1_val_loss_vgg": best_l1["val_loss_vgg"],
            "best_vgg": best_vgg["discriminator"],
            "best_vgg_val_loss_l1": best_vgg["val_loss_l1"],
            "best_vgg_val_loss_vgg": best_vgg["val_loss_vgg"],
            "config": cfg,
            "history": history,
        }, f"{cfg['save_dir']}/discriminator.pt", pickle_module=dill)

        print(f"Best L1 val. loss: {best_l1['val_loss_l1']:.4f} (VGG logg loss: {best_l1['val_loss_vgg']:.4f}) at epoch {best_l1['epoch']}")
        print(f"Best VGG val. loss: {best_vgg['val_loss_vgg']:.4f} (L1 loss: {best_vgg['val_loss_l1']:.4f}) at epoch {best_vgg['epoch']}")
        print(f"Saved models to {cfg['save_dir']}")
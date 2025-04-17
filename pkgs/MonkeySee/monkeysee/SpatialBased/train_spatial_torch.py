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
import numpy as np
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()

from monkeysee.SpatialBased.discriminator import Discriminator
from monkeysee.SpatialBased.generator import Generator
from csng.data import get_dataloaders
from csng.utils.mix import seed_all, check_if_data_zscored
from csng.utils.data import crop
from csng.losses import get_metrics

DATA_PATH_MONKEYSEE = os.path.join(os.environ["DATA_PATH"], "monkeysee")
DATA_PATH_BRAINREADER = os.path.join(os.environ["DATA_PATH"], "brainreader")
DATA_PATH_CAT_V1 = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


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
        "data_name": "mouse_v1",
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
        "target_transforms": {
            "brainreader_mouse": lambda x: x,
            "mouse_v1": lambda x: crop(x, (22, 36)),
            "cat_v1": lambda x: crop(x, (20, 20)),
        },
    },
    # "wandb": None,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "monkeysee",
    },
}

### data
if cfg["data"]["data_name"] == "brainreader_mouse":
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
elif cfg["data"]["data_name"] == "mouse_v1":
    cfg["data"]["mouse_v1"] = {
        "dataset_fn": "sensorium.datasets.static_loaders",
        "dataset_config": {
            "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
                os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-1
                # os.path.join(DATA_PATH_MOUSE_V1, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-2
                # os.path.join(DATA_PATH_MOUSE_V1, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-3
                # os.path.join(DATA_PATH_MOUSE_V1, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-4
                # os.path.join(DATA_PATH_MOUSE_V1, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-5
            ],
            "normalize": True,
            "z_score_responses": True,
            "scale": 0.25, # 256x144 -> 64x36
            "include_behavior": False,
            "add_behavior_as_channels": False,
            "include_eye_position": True,
            "exclude": None,
            "file_tree": True,
            "cuda": "cuda" in cfg["device"],
            "batch_size": 32,
            "drop_last": True,
            "use_cache": False,
        },
        "skip_train": False,
        "skip_val": False,
        "skip_test": False,
        "normalize_neuron_coords": True,
        "average_test_multitrial": True,
        "save_test_multitrial": True,
        "test_batch_size": 7,
        "device": cfg["device"],
    }
elif cfg["data"]["data_name"] == "cat_v1":
    cfg["data"]["cat_v1"] = {
        "dataset_config": {
            "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
            "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
            "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
            "image_size": [50, 50],
            "crop": False,
            "batch_size": 32,
            "stim_keys": ("stim",),
            "resp_keys": ("exc_resp", "inh_resp"),
            "return_coords": True,
            "return_ori": False,
            "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
            "cached": False,
            "stim_normalize_mean": 46.143,
            "stim_normalize_std": 24.960,
            "resp_normalize_mean": torch.load(
                os.path.join(DATA_PATH_CAT_V1, "responses_mean.pt")
            ),
            "resp_normalize_std": torch.load(
                os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
            ),
            "clamp_neg_resp": False,
        },
    }


### model
cfg["decoder"] = {
    "gen": {
        "input_channels": 480,
        "normalized": check_if_data_zscored(cfg=cfg),
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
        "lr": 5e-5,
        "betas": (0.5, 0.999),
        "weight_decay": 1e-3,
    },
    "dis": {
        "input_channels": 1,
        "inp_shape": cfg["data"]["target_transforms"][cfg["data"]["data_name"]](next(iter(get_dataloaders(config=cfg)[0]["train"][cfg["data"]["data_name"]]))[0]["stim"]).shape[1:],
        "lr": 5e-5,
        "betas": (0.5, 0.999),
        "weight_decay": 1e-3,
    },
    "sum_rfs_out": True,
    "standardize_inputs": True,
    "early_stopping_loss_fn": "Alex(5) Loss",
    "early_stopping_eval_for_n_samples": 500, 
    "epochs": 300,
    "load_ckpt": None,
    # "load_ckpt": "/home/jan/Desktop/Dev/MonkeySee/data/models/03-02-2025_00-37"
}
cfg["rfs"] = {
    "meis_path": None,
    # "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis", "6", "meis.pt"),
    # "spatial_embeddings_path": None,
    # "spatial_embeddings_path": os.path.join(
    #     DATA_PATH_MONKEYSEE, "spatial_embedding", "08-02-2025_13-33", "embedding.pt"), # brainreader_mouse (B-6)
    "spatial_embeddings_path": os.path.join(
        DATA_PATH_MONKEYSEE, "spatial_embedding", "23-03-2025_19-07", "embedding_500.pt"), # mouse_v1 (M-1)
    # "spatial_embeddings_path": os.path.join(
    #     DATA_PATH_MONKEYSEE, "spatial_embedding", "22-02-2025_11-53", "embedding.pt"), # cat_v1
    "device": cfg["device"],
}


if __name__ == '__main__':
    print(f"... Running on {cfg['device']} ...")

    assert (
        ("brainreader_mouse" not in cfg["data"] or cfg["data"]["brainreader_mouse"]["sessions"] == [6]) and
        ("mouse_v1" not in cfg["data"] or cfg["data"]["mouse_v1"]["dataset_config"]["paths"] == [os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip")])
    ), "Only B-6 and M-1 are supported now."

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
    wdb_run = None
    if cfg["wandb"] is not None:
        wdb_run = wandb.init(**cfg["wandb"], name=cfg["run_name"], config=cfg, save_code=True,
            tags=["MonkeySee", "Inv-Ret-Map" if cfg["decoder"]["gen"]["inverse_retinotopic_mapping_cfg"] is not None else "MEIs"])
        wdb_run.log_code(".", include_fn=lambda path, root: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") or path.endswith(".yml"))

    ### initialize models
    seed_all(cfg["seed"])
    discriminator = Discriminator(**cfg["decoder"]["dis"], device=cfg["device"])
    generator = Generator(**cfg["decoder"]["gen"], device=cfg["device"])
    if wdb_run is not None:
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
        history = {k: [] for k in ['D_loss_train', 'G_loss_train', 'G_loss_D_train', 'G_loss_vgg_train', 'G_loss_pix_train', 'G_loss_vgg_val', 'G_loss_pix_val', "G_loss_vgg_val", "G_loss_es_val"]}

    ### select early stopping loss function
    es_loss_fn = get_metrics(
        inp_zscored=check_if_data_zscored(cfg=cfg),
        crop_win=None,
        device=cfg["device"],
    )[cfg["decoder"]["early_stopping_loss_fn"]]

    ### collect statistics
    transform_inputs = lambda x: x
    if cfg["decoder"]["standardize_inputs"]:
        print("[INFO] Collecting statistics ...")
        dls, _ = get_dataloaders(config=cfg)
        train_dl = dls["train"][cfg["data"]["data_name"]]
        mean, std = compute_mean_std(dl=train_dl, config=cfg, RFs=RFs)
        print(f"  mean: {mean}\n  std: {std}")
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1)
        transform_inputs = lambda x: (x - mean) / (std + 1e-6)
    transform_targets = cfg["data"]["target_transforms"][cfg["data"]["data_name"]]

    ### train
    best_es = {"epoch": None, "val_loss": np.inf, "recon": None, "generator": None, "discriminator": None}
    best_l1 = {"epoch": None, "val_loss_vgg": np.inf, "val_loss_l1": np.inf, "recon": None, "generator": None, "discriminator": None}
    best_vgg = {"epoch": None, "val_loss_vgg": np.inf, "val_loss_l1": np.inf, "recon": None, "generator": None, "discriminator": None}
    seed_all(cfg["seed"])
    for ep in range(cfg["decoder"]["epochs"]):
        generator.train()
        discriminator.train()
        dls, _ = get_dataloaders(config=cfg)
        train_dl, val_dl = dls["train"][cfg["data"]["data_name"]], dls["val"][cfg["data"]["data_name"]]
        print(
            f"[E {ep+1}/{cfg['decoder']['epochs']}  {datetime.now().strftime('%H:%M:%S')}]\n  "
            + '\n  '.join([f'{k}: {np.mean(v[-len(train_dl):]):.4f}' for k, v in history.items()] if ep > 0 else [])
        )

        for batch in tqdm(train_dl, total=len(train_dl)):
            ### prepare inputs
            brains = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
            targets = transform_targets(torch.cat([dp["stim"] for dp in batch], dim=0)).to(cfg["device"])
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
            vgg_loss, l1_loss, es_loss, n_samples, es_loss_n_samples_counter = 0, 0, 0, 0, 0
            all_targets, all_recons = [], []
            for b_i, batch in enumerate(val_dl):
                brains = torch.cat([dp["resp"] for dp in batch], dim=0).unsqueeze(-1).to(cfg["device"])
                targets = transform_targets(torch.cat([dp["stim"] for dp in batch], dim=0)).to(cfg["device"])
                inputs = get_inputs(brains=brains, config=cfg, transform_inputs_fn=transform_inputs, RFs=RFs)

                recons, inv_ret_maps = generator(inputs, return_inv_ret_maps=True)

                _vgg_loss = generator._lossfun._vgg(recons, targets, reduction="none")
                vgg_loss += torch.stack([vgg_layer_loss.mean(dim=(1, 2, 3)) for vgg_layer_loss in _vgg_loss], dim=1).mean(dim=1).sum().item()
                l1_loss += F.l1_loss(recons, targets, reduction="none").mean(dim=(1, 2, 3)).sum().item()

                all_targets.append(targets.cpu())
                all_recons.append(recons.cpu().detach())
                n_samples += len(recons)
                es_loss_n_samples_counter += len(recons)

                ### eval early stopping loss and average at the end (to prevent OOM)
                if b_i == len(val_dl) - 1 \
                   or (
                        cfg["decoder"]["early_stopping_eval_for_n_samples"] is not None
                        and es_loss_n_samples_counter >= cfg["decoder"]["early_stopping_eval_for_n_samples"]
                    ):
                    es_loss += es_loss_fn(
                        torch.cat(all_recons, dim=0),
                        torch.cat(all_targets, dim=0)
                    ).mean().item() * es_loss_n_samples_counter
                    all_targets, all_recons = [], []
                    es_loss_n_samples_counter = 0

            vgg_loss /= n_samples
            l1_loss /= n_samples
            es_loss /= n_samples

            print(f"  val. VGG: {vgg_loss:.4f}\n  val. L1: {l1_loss:.4f}\n  val. ES: {es_loss:.4f}")
            history["G_loss_vgg_val"].append(vgg_loss)
            history["G_loss_pix_val"].append(l1_loss)
            history["G_loss_es_val"].append(es_loss)

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

            if es_loss < best_es["val_loss"]:
                best_es["epoch"] = ep
                best_es["val_loss"] = es_loss
                best_es["generator"] = deepcopy(generator.state_dict())
                best_es["discriminator"] = deepcopy(discriminator.state_dict())

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
        if best_es["epoch"] == ep:
            fig.savefig(f"{cfg['save_dir']}/recons_best_es.png")
            best_es["recon"] = fig
        plt.close(fig)

        ### log
        if wdb_run is not None:
            wdb_run.log({
                "D_loss_train": np.mean(history["D_loss_train"][-len(train_dl):]),
                "G_loss_train": np.mean(history["G_loss_train"][-len(train_dl):]),
                "G_loss_D_train": np.mean(history["G_loss_D_train"][-len(train_dl):]),
                "G_loss_vgg_train": np.mean(history["G_loss_vgg_train"][-len(train_dl):]),
                "G_loss_vgg_val": history["G_loss_vgg_val"][-1],
                "G_loss_pix_val": history["G_loss_pix_val"][-1],
                "G_loss_es_val": history["G_loss_es_val"][-1],
                "reconstructions": fig,
                "best_l1_reconstructions": best_l1["recon"],
                "best_vgg_reconstructions": best_vgg["recon"],
                "best_es_reconstructions": best_es["recon"],
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
            "best_es": best_es["generator"],
            "best_es_val_loss": best_es["val_loss"],
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
            "best_es": best_es["discriminator"],
            "best_es_val_loss": best_es["val_loss"],
            "config": cfg,
            "history": history,
        }, f"{cfg['save_dir']}/discriminator.pt", pickle_module=dill)

        print(f"Best L1 val. loss: {best_l1['val_loss_l1']:.4f} (VGG logg loss: {best_l1['val_loss_vgg']:.4f}) at epoch {best_l1['epoch']}")
        print(f"Best VGG val. loss: {best_vgg['val_loss_vgg']:.4f} (L1 loss: {best_vgg['val_loss_l1']:.4f}) at epoch {best_vgg['epoch']}")
        print(f"Best ES val. loss: {best_es['val_loss']:.4f} at epoch {best_es['epoch']}")
        print(f"Saved models to {cfg['save_dir']}")
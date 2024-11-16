import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import lovely_tensors as lt
lt.monkey_patch()

from csng.models.cnn import CNN
from csng.utils.mix import seed_all, plot_losses, plot_comparison, count_parameters
from csng.utils.data import crop, standardize, normalize
from csng.losses import SSIMLoss, Loss, get_metrics
from csng.models.readins import (
    MultiReadIn,
    ConvReadIn,
    FCReadIn,
    MEIReadIn,
)
from csng.data import get_dataloaders, get_sample_data
from csng.models.utils.cnn import init_decoder, setup_run_dir, setup_wandb_run, train, val

### set paths
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")




##### global run config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "save_run": True,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "cnn_decoder",
    },
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "crop_wins": dict(),
}

### brainreader mouse data
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 6,
    # "sessions": list(range(1, 3)),
    "sessions": [6],
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}

### cat v1 data
config["data"]["cat_v1"] = {
    "crop_win": (20, 20),
    "dataset_config": {
        "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
        "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
        "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
        "image_size": [50, 50],
        "crop": False,
        "batch_size": 6,
        "stim_keys": ("stim",),
        "resp_keys": ("exc_resp", "inh_resp"),
        "return_coords": True,
        "return_ori": False,
        "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
        "cached": False,
        "stim_normalize_mean": 46.143,
        "stim_normalize_std": 20.420,
        "resp_normalize_mean": torch.load(
            os.path.join(DATA_PATH_CAT_V1, "responses_mean.pt")
        ),
        "resp_normalize_std": torch.load(
            os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
        ),
        # "training_sample_idxs": np.random.choice(45000, size=22330, replace=False),
    },
}

### mouse v1 data
config["data"]["mouse_v1"] = {
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
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 6,
        "seed": config["seed"],
        "use_cache": False,
    },
    "crop_win": (22, 36),
    "skip_train": False,
    "skip_val": False,
    "skip_test": False,
    "normalize_neuron_coords": True,
    "average_test_multitrial": True,
    "save_test_multitrial": True,
    "test_batch_size": 7,
    "device": config["device"],
}

### decoder
config["decoder"] = {
    "readin_type": (readin_type := "mei"), # "conv", "fc", "mei"
    "model": {
        "readins_config": [], # specified below
        "core_cls": CNN,
        "core_config": {
            "layers": {
                "conv": [
                    ("deconv", 480, 7, 2, 2),
                    ("deconv", 256, 5, 1, 2),
                    ("deconv", 256, 5, 1, 2),
                    ("deconv", 128, 4, 1, 1),
                    ("deconv", 64, 3, 1, 1),
                    ("deconv", 1, 3, 1, 0),
                ],
                "fc": [
                    ("deconv", 480, 7, 2, 2),
                    ("deconv", 256, 5, 1, 2),
                    ("deconv", 256, 5, 1, 2),
                    ("deconv", 128, 4, 1, 1),
                    ("deconv", 64, 3, 1, 1),
                    ("deconv", 1, 3, 1, 0),
                ],
                "mei": [
                    ("conv", 480, 7, 1, 3),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 128, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    ("conv", 1, 3, 1, 1),
                ],
            }[readin_type],
            "act_fn": nn.ReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.35,
            "batch_norm": True,
        },
    },
    "opter_cls": torch.optim.AdamW,
    "opter_kwargs": {"lr": 3e-4, "weight_decay": 0.03},
    "loss": {
        "loss_fn": dict(),
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
    },
    "val_loss": "FID", # get_metrics(config)["SSIML-PL"],
    "n_epochs": 100,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_only_core": False,
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-08-22_23-08-20", "ckpt", "decoder_185.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "qu6wnt7h",
    # },
}

### finish config for brainreader mouse
if "brainreader_mouse" in config["data"]:
    _dls, _ = get_dataloaders(config=config)
    for data_key, dset in zip(_dls["train"]["brainreader_mouse"].data_keys, _dls["train"]["brainreader_mouse"].datasets):
        ### set crop wins and losses
        config["crop_wins"][data_key] = tuple(dset[0].images.shape[-2:])
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["crop_wins"][data_key], log_loss=True, inp_normalized=True, inp_standardized=False)

        ### append readin
        n_neurons = dset[0].responses.shape[-1]
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_neurons,
            "decoding_objective_config": None,
            "layers": {
                "conv": [
                    (ConvReadIn, {
                        "H": 18,
                        "W": 32,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 1, # resp
                            "layers_config": [("fc", 8), ("fc", 64), ("fc", 18*32)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.2,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": n_neurons,
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.2,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),
                ],
                "fc": [
                    (FCReadIn, {
                        "in_shape": n_neurons,
                        "layers_config": [
                            ("fc", 1728),
                            ("unflatten", 1, (3, 18, 32)),
                        ],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "batch_norm": True,
                        "dropout": 0.15,
                    }),
                ],
                "mei": [
                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH_BRAINREADER, "meis", data_key,  "meis.pt"),
                        "n_neurons": n_neurons,
                        "mei_resize_method": "resize",
                        "mei_target_shape": (36, 64),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.15,
                        },
                        "ctx_net_config": {
                            "in_channels": 1, # resp, x, y
                            "layers_config": [("fc", 8), ("fc", 128), ("fc", 36*64)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.15,
                            "batch_norm": True,
                        },
                        "shift_coords": False,
                        "device": config["device"],
                    }),
                ],
            }[config["decoder"]["readin_type"]],
        })

### finish config for cat v1
if "cat_v1" in config["data"]:
    ### set crop wins and losses
    config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]
    config["decoder"]["loss"]["loss_fn"]["cat_v1"] = SSIMLoss(window=config["data"]["cat_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)

    ### append readin
    config["decoder"]["model"]["readins_config"].append({
        "data_key": "cat_v1",
        "in_shape": 46875,
        "decoding_objective_config": None,
        "layers": {
            "conv": [
                (ConvReadIn, {
                    "H": 8,
                    "W": 8,
                    "shift_coords": False,
                    "learn_grid": True,
                    "grid_l1_reg": 8e-3,
                    "in_channels_group_size": 1,
                    "grid_net_config": {
                        "in_channels": 3, # x, y, resp
                        "layers_config": [("fc", 64), ("fc", 128), ("fc", 8*8)],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "dropout": 0.15,
                        "batch_norm": False,
                    },
                    "pointwise_conv_config": {
                        "in_channels": 46875,
                        "out_channels": 480,
                        "act_fn": nn.Identity,
                        "bias": False,
                        "batch_norm": True,
                        "dropout": 0.1,
                    },
                    "gauss_blur": False,
                    "gauss_blur_kernel_size": 7,
                    "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                    # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
                    "gauss_blur_sigma_init": 1.5,
                    "neuron_emb_dim": None,
                }),
            ],
            "fc": [
                (FCReadIn, {
                    "in_shape": 46875,
                    "layers_config": [
                        ("fc", 192),
                        ("unflatten", 1, (3, 8, 8)),
                    ],
                    "act_fn": nn.LeakyReLU,
                    "out_act_fn": nn.Identity,
                    "batch_norm": True,
                    "dropout": 0.15,
                    "out_channels": 8,
                }),
            ],
            "mei": [
                (MEIReadIn, {
                    "meis_path": os.path.join(DATA_PATH_CAT_V1, "meis", "cat_v1",  "meis.pt"),
                    "n_neurons": 46875,
                    "mei_resize_method": "resize",
                    "mei_target_shape": (20, 20),
                    "pointwise_conv_config": {
                        "out_channels": 480,
                        "bias": False,
                        "batch_norm": True,
                        "act_fn": nn.LeakyReLU,
                        "dropout": 0.15,
                    },
                    "ctx_net_config": {
                        "in_channels": 3, # resp, x, y
                        "layers_config": [("fc", 64), ("fc", 128), ("fc", 20*20)],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "dropout": 0.15,
                        "batch_norm": True,
                    },
                    "shift_coords": False,
                    "device": config["device"],
                }),
            ],
        }[config["decoder"]["readin_type"]],
    })

### finish config for mouse v1
if "mouse_v1" in config["data"]:
    _dls, _neuron_coords = get_dataloaders(config=config)
    for data_key, n_coords in _dls["train"]["mouse_v1"].neuron_coords.items():
        ### set crop wins and losses
        config["crop_wins"][data_key] = config["data"]["mouse_v1"]["crop_win"]
        config["decoder"]["loss"]["loss_fn"][data_key] = SSIMLoss(window=config["data"]["mouse_v1"]["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False)

        ### append readin
        config["decoder"]["model"]["readins_config"].append({
            "data_key": data_key,
            "in_shape": n_coords.shape[-2],
            "decoding_objective_config": None,
            "layers": {
                "conv": [
                    (ConvReadIn, {
                        "H": 10,
                        "W": 18,
                        "shift_coords": False,
                        "learn_grid": True,
                        "grid_l1_reg": 8e-3,
                        "in_channels_group_size": 1,
                        "grid_net_config": {
                            "in_channels": 3, # x, y, resp
                            "layers_config": [("fc", 32), ("fc", 86), ("fc", 18*10)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.1,
                            "batch_norm": False,
                        },
                        "pointwise_conv_config": {
                            "in_channels": n_coords.shape[-2],
                            "out_channels": 480,
                            "act_fn": nn.Identity,
                            "bias": False,
                            "batch_norm": True,
                            "dropout": 0.1,
                        },
                        "gauss_blur": False,
                        "gauss_blur_kernel_size": 7,
                        "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
                    }),
                ],
                "fc": [
                    (FCReadIn, {
                        "in_shape": n_coords.shape[-2],
                        "layers_config": [
                            ("fc", 540),
                            ("unflatten", 1, (3, 10, 18)),
                        ],
                        "act_fn": nn.LeakyReLU,
                        "out_act_fn": nn.Identity,
                        "batch_norm": True,
                        "dropout": 0.15,
                    }),
                ],
                "mei": [
                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH_MOUSE_V1, "meis", data_key,  "meis.pt"),
                        "n_neurons": n_coords.shape[-2],
                        "mei_resize_method": "resize",
                        "mei_target_shape": (22, 36),
                        "pointwise_conv_config": {
                            "out_channels": 480,
                            "bias": False,
                            "batch_norm": True,
                            "act_fn": nn.LeakyReLU,
                            "dropout": 0.1,
                        },
                        "ctx_net_config": {
                            "in_channels": 3, # resp, x, y
                            "layers_config": [("fc", 32), ("fc", 128), ("fc", 22*36)],
                            "act_fn": nn.LeakyReLU,
                            "out_act_fn": nn.Identity,
                            "dropout": 0.1,
                            "batch_norm": True,
                        },
                        "shift_coords": False,
                        "device": config["device"],
                    }),
                ],
            }[config["decoder"]["readin_type"]],
        })



### main pipeline
def run(cfg):
    print(f"... Running on {cfg['device']} ...")
    print(f"{DATA_PATH=}")
    seed_all(cfg["seed"])

    ### get data sample for plotting and logging
    dls, neuron_coords = get_dataloaders(config=cfg)
    s = get_sample_data(dls=dls, config=cfg)
    resp, sample_dataset, sample_data_key = s["resp"], s["sample_dataset"], s["sample_data_key"]

    ### init decoder (and load ckpt if needed)
    cfg, decoder, opter, loss_fn, history, best, ckpt = init_decoder(config=cfg)
    with torch.no_grad():
        print(
            decoder,
            f"\n\n-----"
            f"\nOutput shape: {decoder(resp.to(cfg['device']), data_key=sample_data_key, neuron_coords=neuron_coords[sample_dataset][sample_data_key]).shape}"
            f"\n-----"
            f"\nNumber of parameters:"
            f"\n  whole model: {count_parameters(decoder)}"
            f"\n  core: {count_parameters(decoder.core)} ({count_parameters(decoder.core) / count_parameters(decoder) * 100:.2f}%)"
            f"\n  readins: {count_parameters(decoder.readins)} ({count_parameters(decoder.readins) / count_parameters(decoder) * 100:.2f}%)"
            f"\n    ({', '.join([f'{k}: {count_parameters(v)} [{count_parameters(v) / count_parameters(decoder) * 100:.2f}%]' for k, v in decoder.readins.items()])})"
        )

    ### prepare run name and run directory
    cfg, make_sample_path = setup_run_dir(config=cfg, ckpt=ckpt)

    ### prepare wandb logging
    wdb_run = setup_wandb_run(config=cfg, decoder=decoder)

    ### setup (e)val loss
    val_loss_fn = cfg["decoder"]["val_loss"] or Loss(model=decoder, config=cfg["decoder"]["loss"])

    ### train
    print(f"[INFO] cfg:\n{json.dumps(cfg, indent=2, default=str)}")
    start, end = len(history["train_loss"]), cfg["decoder"]["n_epochs"]
    for epoch in range(start, end):
        print(f"[{epoch}/{end}]")

        ### train and val
        dls, neuron_coords = get_dataloaders(config=cfg)
        train_loss = train(
            model=decoder,
            dataloaders=dls["train"],
            opter=opter,
            loss_fn=loss_fn,
            config=cfg,
        )
        val_loss = val(
            model=decoder,
            dataloaders=dls["val"],
            loss_fn=val_loss_fn,
            config=cfg,
        )

        ### save best model
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = epoch
            best["model"] = deepcopy(decoder.state_dict())

        ### log
        print(f"{train_loss=:.4f}, {val_loss=:.4f}")
        if cfg["wandb"]: wdb_run.log({"train_loss": train_loss, "val_loss": val_loss}, commit=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        ### plot reconstructions
        if "brainreader_mouse" in cfg["data"]:
            b_stim_pred = decoder(s["b_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["b_stim"][:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(), pred=crop(b_stim_pred[:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "b_"), show=False)
        if "cat_v1" in cfg["data"]:
            c_stim_pred = decoder(s["c_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]], data_key=s["c_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["c_stim"][:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(), pred=crop(c_stim_pred[:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "c_"), show=False)
        if "mouse_v1" in cfg["data"]:
            m_stim_pred = decoder(s["m_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"][:8].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach()
            fig = plot_comparison(target=crop(s["m_stim"][:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(), pred=crop(m_stim_pred[:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(), save_to=make_sample_path(epoch, "m_"), show=False)
        if cfg["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(cfg["dir"], f"losses_{epoch}.png") if cfg["save_run"] else None)

        ### save ckpt
        if cfg["save_run"]:
            torch.save({
                "decoder": decoder.state_dict(),
                "opter": opter.state_dict(),
                "history": history,
                "config": cfg,
                "best": best,
            }, os.path.join(cfg["dir"], "ckpt", f"decoder_{epoch}.pt"), pickle_module=dill)

    ### final evaluation + logging + saving
    print("\n\n-----\n" + f"Best val loss: {best['val_loss']:.4f} at epoch {best['epoch']}")

    ### save final ckpt
    if cfg["save_run"]:
        torch.save({
            "decoder": decoder.state_dict(),
            "opter": opter.state_dict(),
            "history": history,
            "config": cfg,
            "best": best,
        }, os.path.join(cfg["dir"], f"decoder.pt"), pickle_module=dill)

    ### eval on test set w/ current params
    print("Evaluating on test set with current model...")
    dls, neuron_coords = get_dataloaders(config=cfg)
    curr_test_loss = val(
        model=decoder,
        dataloaders=dls["test"],
        loss_fn=val_loss_fn,
        config=cfg,
    )
    print(f"  Test loss (current model): {curr_test_loss:.4f}")

    ### load best model
    decoder.load_state_dict(best["model"])

    ### eval on test set w/ best params
    print("Evaluating on test set with the best model...")
    dls, neuron_coords = get_dataloaders(config=cfg)
    final_test_loss = val(
        model=decoder,
        dataloaders=dls["test"],
        loss_fn=val_loss_fn,
        config=cfg,
    )
    print(f"  Test loss (best model): {final_test_loss:.4f}")

    ### plot reconstructions of the final model
    if "brainreader_mouse" in cfg["data"]:
        b_stim_pred_best = decoder(s["b_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["b_stim"][:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(),
            pred=crop(b_stim_pred_best[:8], cfg["crop_wins"][s["b_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "b_stim_comparison_best.png") if cfg["save_run"] else None,
        )
    if "cat_v1" in cfg["data"]:
        c_stim_pred_best = decoder(s["c_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]], data_key=s["c_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["c_stim"][:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(),
            pred=crop(c_stim_pred_best[:8], cfg["crop_wins"][s["c_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "c_stim_comparison_best.png") if cfg["save_run"] else None,
        )
    if "mouse_v1" in cfg["data"]:
        m_stim_pred_best = decoder(s["m_resp"][:8].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"][:8].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach().cpu()
        fig = plot_comparison(
            target=crop(s["m_stim"][:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(),
            pred=crop(m_stim_pred_best[:8], cfg["crop_wins"][s["m_sample_data_key"]]).cpu(),
            show=False,
            save_to=os.path.join(cfg["dir"], "m_stim_comparison_best.png") if cfg["save_run"] else None,
        )

    ### finish wandb run
    if cfg["wandb"]:
        wandb.run.summary["best_val_loss"] = best["val_loss"]
        wandb.run.summary["best_epoch"] = best["epoch"]
        wandb.run.summary["curr_test_loss"] = curr_test_loss
        wandb.run.summary["final_test_loss"] = final_test_loss
        wandb.run.summary["best_reconstruction"] = fig
        wdb_run.finish()

    ### plot losses
    plot_losses(
        history=history,
        show=False,
        save_to=None if not cfg["save_run"] else os.path.join(cfg["dir"], f"losses.png"),
    )


if __name__ == "__main__":
    run(cfg=config)

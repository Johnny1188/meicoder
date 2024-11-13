import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt
import wandb

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import plot_losses, plot_comparison, standardize, normalize, count_parameters, crop
from csng.losses import SSIMLoss, Loss, CroppedLoss
from csng.readins import (
    MultiReadIn,
    HypernetReadIn,
    ConvReadIn,
    AttentionReadIn,
    FCReadIn,
    AutoEncoderReadIn,
    Conv1dReadIn,
    LocalizedFCReadIn,
    MEIReadIn,
)
from csng.GAN import GAN

from csng.cat_v1.encoder import get_encoder
from csng.cat_v1.gan_utils import (
    train,
    val,
    get_dataloaders,
)
from csng.cat_v1.data import (
    prepare_v1_dataloaders,
    SyntheticDataset,
    BatchPatchesDataLoader,
    MixedBatchLoader,
    PerSampleStoredDataset,
)

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")



##### set run config #####
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
    },
    # "crop_win": (slice(15, 35), slice(15, 35)),
    "crop_win": (20, 20),
    "only_cat_v1_eval": True,
    "device": os.environ["DEVICE"],
    "seed": 0,
    "wandb": None,
    "wandb": {
        "project": os.environ["WANDB_PROJECT"],
        "group": "cat_v1",
    },
}
config["data"]["cat_v1"] = {
    "train_path": os.path.join(DATA_PATH, "datasets", "train"),
    "val_path": os.path.join(DATA_PATH, "datasets", "val"),
    "test_path": os.path.join(DATA_PATH, "datasets", "test"),
    "image_size": [50, 50],
    "crop": False,
    "batch_size": 64,
    "stim_keys": ("stim",),
    "resp_keys": ("exc_resp", "inh_resp"),
    "return_coords": True,
    "return_ori": False,
    "coords_ori_filepath": os.path.join(DATA_PATH, "pos_and_ori.pkl"),
    "cached": False,
    "stim_normalize_mean": 46.143,
    "stim_normalize_std": 20.420,
    "resp_normalize_mean": torch.load(
        os.path.join(DATA_PATH, "responses_mean.pt")
    ),
    "resp_normalize_std": torch.load(
        os.path.join(DATA_PATH, "responses_std.pt")
    ),
}

config["decoder"] = {
    "model": {
        "readins_config": [
            {
                "data_key": "cat_v1",
                "in_shape": 46875,
                "decoding_objective_config": None,
                "layers": [
                    # (ConvReadIn, {
                    #     "H": 8,
                    #     "W": 8,
                    #     "shift_coords": False,
                    #     "learn_grid": True,
                    #     "grid_l1_reg": 8e-3,
                    #     "in_channels_group_size": 1,
                    #     "grid_net_config": {
                    #         "in_channels": 3, # x, y, resp
                    #         "layers_config": [("fc", 64), ("fc", 128), ("fc", 8*8)],
                    #         "act_fn": nn.LeakyReLU,
                    #         "out_act_fn": nn.Identity,
                    #         "dropout": 0.15,
                    #         "batch_norm": False,
                    #     },
                    #     "pointwise_conv_config": {
                    #         "in_channels": 46875,
                    #         "out_channels": 480,
                    #         "act_fn": nn.Identity,
                    #         "bias": False,
                    #         "batch_norm": True,
                    #         "dropout": 0.1,
                    #     },
                    #     "gauss_blur": False,
                    #     "gauss_blur_kernel_size": 7,
                    #     "gauss_blur_sigma": "fixed", # "fixed", "single", "per_neuron"
                    #     # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
                    #     "gauss_blur_sigma_init": 1.5,
                    #     "neuron_emb_dim": None,
                    # }),

                    # (FCReadIn, {
                    #     "in_shape": 46875,
                    #     "layers_config": [
                    #         ("fc", 512),
                    #         ("unflatten", 1, (8, 8, 8)),
                    #     ],
                    #     "act_fn": nn.LeakyReLU,
                    #     "out_act_fn": nn.Identity,
                    #     "batch_norm": True,
                    #     "dropout": 0.15,
                    #     "out_channels": 8,
                    # }),

                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH, "meis", "cat_v1",  "meis.pt"),
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
            }
        ],
        "core_cls": GAN,
        "core_config": {
            "G_kwargs": {
                "in_shape": [480],
                "layers": [
                    # ("deconv", 480, 7, 2, 3),
                    # ("deconv", 256, 5, 1, 2),
                    # ("deconv", 256, 5, 1, 1),
                    # ("deconv", 128, 4, 1, 1),
                    # ("deconv", 64, 3, 1, 1),
                    # ("deconv", 1, 3, 1, 0),

                    ### MEI readin
                    ("conv", 480, 7, 1, 3),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 128, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    ("conv", 1, 3, 1, 1),
                ],
                "act_fn": nn.ReLU,
                "out_act_fn": nn.Identity,
                "dropout": 0.35,
                "batch_norm": True,
            },
            "D_kwargs": {
                "in_shape": [1, 20, 20],
                "layers": [
                    # ("conv", 128, 4, 1, 2),
                    # ("conv", 128, 4, 1, 0),
                    # ("conv", 64, 4, 1, 1),
                    # ("conv", 32, 3, 1, 0),
                    # ("fc", 1),

                    ("conv", 256, 7, 2, 2),
                    ("conv", 256, 5, 1, 2),
                    ("conv", 128, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    ("conv", 64, 3, 1, 1),
                    ("fc", 1),
                ],
                "act_fn": nn.ReLU,
                "out_act_fn": nn.Sigmoid,
                "dropout": 0.3,
                "batch_norm": True,
            },
        },
    },
    "loss": {
        # "loss_fn": CroppedLoss(window=config["crop_win"], loss_fn=nn.MSELoss(), normalize=False, standardize=False),
        "loss_fn": SSIMLoss(
            window=config["crop_win"],
            log_loss=True,
            inp_normalized=True,
            inp_standardized=False,
        ),
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
        # "con_reg_mul": 0,
        "con_reg_mul": 1,
        "con_reg_loss_fn": SSIMLoss(window=config["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        # "encoder": None,
        "encoder": get_encoder(
            device=config["device"],
            eval_mode=True,
            ckpt_path=os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter.pth"),
        ),
    },
    "G_opter_cls": torch.optim.AdamW,
    "G_opter_kwargs": {"lr": 3e-4, "weight_decay": 0.03},
    "D_opter_cls": torch.optim.AdamW,
    "D_opter_kwargs": {"lr": 3e-4, "weight_decay": 0.03},
    "G_reg": {"l1": 0, "l2": 0},
    "D_reg": {"l1": 0, "l2": 0},
    "G_adv_loss_mul": 0.1,
    "G_stim_loss_mul": 0.9,
    "D_real_loss_mul": 0.5,
    "D_fake_loss_mul": 0.5,
    "D_real_stim_labels_noise": 0.05,
    "D_fake_stim_labels_noise": 0.05,
    "n_epochs": 70,
    "load_ckpt": None,
    "load_ckpt": {
        "load_best": False,
        "load_opter_state": True,
        "reset_history": False,
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-26_22-18-25", "ckpt", "decoder_50.pt"),
        # "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-24_09-36-46", "decoder.pt"),
        "resume_checkpointing": True,
        "resume_wandb_id": "ch4e0bn6",
    },
    "save_run": True,
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### data
    dataloaders = dict()
    dataloaders["cat_v1"] = prepare_v1_dataloaders(**config["data"]["cat_v1"])

    ### sample data
    sample_data_key = "cat_v1"
    datapoint = next(iter(dataloaders["cat_v1"]["test"]))
    stim, resp, neuron_coords = datapoint.images, datapoint.responses, datapoint.neuron_coords.to(config["device"])

    ### decoder
    ### initialize (and load ckpt if needed)
    if config["decoder"]["load_ckpt"] != None:
        print(f"[INFO] Loading checkpoint from {config['decoder']['load_ckpt']['ckpt_path']}...")
        ckpt = torch.load(config["decoder"]["load_ckpt"]["ckpt_path"], map_location=config["device"], pickle_module=dill)

        history = ckpt["history"]
        config["decoder"]["model"] = ckpt["config"]["decoder"]["model"]
        best = ckpt["best"]

        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        if config["decoder"]["load_ckpt"]["load_best"]:
            core_state_dict = {".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "G" in k or "D" in k}
        else:
            core_state_dict = {".".join(k.split(".")[1:]):v for k,v in ckpt["decoder"].items() if "G" in k or "D" in k}
        decoder.core.G.load_state_dict(core_state_dict["G"])
        decoder.core.D.load_state_dict(core_state_dict["D"])
        if config["decoder"]["load_ckpt"]["load_best"]:
            decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "readin" in k})
        else:
            decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in ckpt["decoder"].items() if "readin" in k})
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])
        if config["decoder"]["load_ckpt"]["load_opter_state"]:
            decoder.core.G_optim.load_state_dict(core_state_dict["G_optim"])
            decoder.core.D_optim.load_state_dict(core_state_dict["D_optim"])

        if config["decoder"]["load_ckpt"]["reset_history"]:
            history = {"val_loss": []}

        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
    else:
        print("[INFO] Initializing the model from scratch...")
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        decoder.core.G_optim = config["decoder"]["G_opter_cls"]([*decoder.core.G.parameters(), *decoder.readins.parameters()], **config["decoder"]["G_opter_kwargs"])
        decoder.core.D_optim = config["decoder"]["D_opter_cls"](decoder.core.D.parameters(), **config["decoder"]["D_opter_kwargs"])
        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])

        history = {"val_loss": []}
        best = {"val_loss": np.inf, "epoch": 0, "model": None}

    ### print model and fix sizes of stimuli
    with torch.no_grad():
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords)
        print(stim_pred.shape)
        del stim_pred

    print(
        decoder,
        "\n---\n"
        f"Number of parameters:"
        f"\n\tTotal: {count_parameters(decoder)}"
        f"\n\tG: {count_parameters(decoder.core.G)}"
        f"\n\tD: {count_parameters(decoder.core.D)}"
    )

    ### prepare checkpointing and wandb logging
    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["decoder"]["save_run"]:
            ### save config
            config["dir"] = os.path.join(DATA_PATH, "models", "gan", config["run_name"])
            os.makedirs(config["dir"], exist_ok=True)
            with open(os.path.join(config["dir"], "config.json"), "w") as f:
                json.dump(config, f, indent=4, default=str)
            os.makedirs(os.path.join(config["dir"], "samples"), exist_ok=True)
            os.makedirs(os.path.join(config["dir"], "ckpt"), exist_ok=True)
            make_sample_path = lambda epoch, prefix: os.path.join(
                config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
            )
            print(f"Run name: {config['run_name']}\nRun dir: {config['dir']}")
        else:
            make_sample_path = lambda epoch, prefix: None
            print("[WARNING] Not saving the run and the config.")
    else:
        config["run_name"] = ckpt["config"]["run_name"]
        config["dir"] = ckpt["config"]["dir"]
        make_sample_path = lambda epoch, prefix: os.path.join(
            config["dir"], "samples", f"{prefix}stim_comparison_{epoch}e.png"
        )
        print(f"Checkpointing resumed - Run name: {config['run_name']}\nRun dir: {config['dir']}")

    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_wandb_id"] == None:
        if config["wandb"]:
            wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config,
                tags=[
                    config["decoder"]["model"]["core_cls"].__name__,
                    config["decoder"]["model"]["readins_config"][0]["layers"][0][0].__name__,
                ],
                notes=None)
            wdb_run.watch(decoder)
        else:
            print("[WARNING] Not using wandb.")
            wdb_run = None
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must")

    ### train
    print(f"[INFO] Config:\n{json.dumps(config, indent=2, default=str)}")
    s, e = len(history["val_loss"]), config["decoder"]["n_epochs"]
    for epoch in range(s, e):
        print(f"[{epoch}/{e}]")

        ### train and val
        train_dataloader, val_dataloader, _ = get_dataloaders(
            config=config,
            dataloaders=dataloaders,
            only_cat_v1_eval=config["only_cat_v1_eval"],
        )
        history = train(
            model=decoder,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            config=config,
            history=history,
            wdb_run=wdb_run,
            wdb_commit=False,
        )
        val_losses = val(
            model=decoder,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
        )

        ### save best model
        if val_losses["total"] < best["val_loss"]:
            best["val_loss"] = val_losses["total"]
            best["epoch"] = epoch
            best["model"] = deepcopy(decoder.state_dict())

        ### log
        history["val_loss"].append(val_losses["total"])
        print(f"Validation loss={val_losses['total']:.4f}")
        if config["wandb"]: wdb_run.log({"val_loss": val_losses["total"]}, commit=False)

        ### plot reconstructions
        stim_pred = decoder(resp[:8].to(config["device"]), neuron_coords=neuron_coords[:8].to(config["device"]), data_key="cat_v1").detach()
        fig = plot_comparison(target=crop(stim[:8], config["crop_win"]).cpu(), pred=crop(stim_pred[:8], config["crop_win"]).cpu(), save_to=make_sample_path(epoch, ""))
        if config["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(config["dir"], f"losses_{epoch}.png") if config["decoder"]["save_run"] else None)

        ### save ckpt
        if epoch % 5 == 0 and epoch > 0:
            ### ckpt
            if config["decoder"]["save_run"]:
                torch.save({
                    "decoder": decoder.state_dict(),
                    "history": history,
                    "config": config,
                    "best": best,
                }, os.path.join(config["dir"], "ckpt", f"decoder_{epoch}.pt"), pickle_module=dill)

    ### final evaluation + logging + saving
    print(f"Best val loss: {best['val_loss']:.4f} at epoch {best['epoch']}")

    ### save final ckpt
    if config["decoder"]["save_run"]:
        torch.save({
            "decoder": decoder.state_dict(),
            "history": history,
            "config": config,
            "best": best,
        }, os.path.join(config["dir"], f"decoder.pt"), pickle_module=dill)

    ### eval on test set w/ current params
    print("Evaluating on test set with current model...")
    _, _, test_dataloader = get_dataloaders(
        config=config,
        dataloaders=dataloaders,
        only_cat_v1_eval=config["only_cat_v1_eval"],
    )
    test_loss_curr = val(
        model=decoder,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
    )
    print(f"  Test loss (current model): {test_loss_curr['total']:.4f}")

    ### load best model
    decoder.core.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "G" in k or "D" in k})
    decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "readin" in k})

    ### eval on test set w/ best params
    print("Evaluating on test set with best model...")
    _, _, test_dataloader = get_dataloaders(
        config=config,
        dataloaders=dataloaders,
        only_cat_v1_eval=config["only_cat_v1_eval"],
    )
    test_loss_final = val(
        model=decoder,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
    )
    print(f"  Test loss (best model): {test_loss_final['total']:.4f}")

    ### plot reconstructions of the final model
    stim_pred_best = decoder(
        resp.to(config["device"]),
        data_key=sample_data_key,
        neuron_coords=neuron_coords,
    ).detach().cpu()
    fig = plot_comparison(
        target=crop(stim[:8], config["crop_win"]).cpu(),
        pred=crop(stim_pred_best[:8], config["crop_win"]).cpu(),
        save_to=os.path.join(config["dir"], "stim_comparison_best.png") if config["decoder"]["save_run"] else None,
        show=False,
    )

    ### log
    if config["wandb"]:
        wandb.run.summary["best_val_loss"] = best["val_loss"]
        wandb.run.summary["best_epoch"] = best["epoch"]
        wandb.run.summary["curr_test_loss"] = test_loss_curr["total"]
        wandb.run.summary["final_test_loss"] = test_loss_final["total"]
        wandb.run.summary["best_reconstruction"] = fig

    ### save/delete wandb run
    if config["wandb"]:
        print("Finishing wandb run...")
        wdb_run.finish()

    ### plot losses
    plot_losses(
        history=history,
        show=False,
        save_to=None if not config["decoder"]["save_run"] else os.path.join(config["dir"], f"losses.png"),
    )

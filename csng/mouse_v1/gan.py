import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import dill
from datetime import datetime
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import lovely_tensors as lt
import wandb

import csng
from csng.GAN import GAN
from csng.utils import crop, plot_comparison, standardize, normalize, plot_losses, count_parameters
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
from csng.comparison import get_metrics

from encoder import get_encoder
from gan_utils import train, val, get_all_data

lt.monkey_patch()
wandb.login()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


##### set run config #####
config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "mouse_v1": None,
        "syn_dataset_config": None,
        "data_augmentation": None,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "crop_win": (22, 36),
    "wandb": None,
    "wandb": {
        "project": "CSNG",
        "group": "sensorium_2022",
    },
}
config["data"]["mouse_v1"] = {
    "dataset_fn": "sensorium.datasets.static_loaders",
    "dataset_config": {
        "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
            # os.path.join(DATA_PATH, "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
            # os.path.join(DATA_PATH, "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
            os.path.join(DATA_PATH, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
            # os.path.join(DATA_PATH, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
            # os.path.join(DATA_PATH, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
            # os.path.join(DATA_PATH, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
            # os.path.join(DATA_PATH, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
        ],
        "normalize": True,
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 16,
        "seed": config["seed"],
        "use_cache": False,
    },
    "skip_train": False,
    "skip_val": False,
    "skip_test": False,
    "normalize_neuron_coords": True,
    "average_test_multitrial": True,
    "save_test_multitrial": True,
    "test_batch_size": 7,
    "device": config["device"],
}

# config["data"]["syn_dataset_config"] = {
#     "data_keys": [
#         "21067-10-18",
#         "22846-10-16",
#         "23343-5-17",
#         "23656-14-22",
#         "23964-4-22",
#     ],
#     "batch_size": 7,
#     "append_data_parts": ["train"],
#     # "data_key_prefix": "syn",
#     "data_key_prefix": None, # the same data key as the original (real) data
#     "dir_name": "synthetic_data_mouse_v1_encoder_new_stimuli",
#     "device": config["device"],
# }
_dataloaders, _ = get_all_data(config=config)

# config["data"]["data_augmentation"] = {
#     "data_transforms": [[  # for synthetic data
#         RespGaussianNoise(
#             noise_std=1.5 * torch.from_numpy(np.load(os.path.join(DATA_PATH, dataset.dirname, "stats", f"responses_iqr.npy"))).float().to(config["device"]),
#             clip_min=0.0,
#             dynamic_mul_factor=0.08,
#             resp_fn="squared",
#         ) for dataset in _dataloaders["mouse_v1"]["train"].datasets
#     ]],
#     "append_data_parts": ["train"],
#     "force_same_order": True,
#     "seed": config["seed"],
# }
_dataloaders, _ = get_all_data(config=config)

config["decoder"] = {
    "model": {
        "readins_config": [
            {
                "data_key": data_key,
                "in_shape": n_coords.shape[-2],
                "decoding_objective_config": None,
                "layers": [
                    # (ConvReadIn, {
                    #     "H": 10,
                    #     "W": 18,
                    #     "shift_coords": False,
                    #     "learn_grid": True,
                    #     "grid_l1_reg": 8e-3,
                    #     "in_channels_group_size": 1,
                    #     "grid_net_config": {
                    #         "in_channels": 3, # x, y, resp
                    #         "layers_config": [("fc", 32), ("fc", 86), ("fc", 18*10)],
                    #         "act_fn": nn.LeakyReLU,
                    #         "out_act_fn": nn.Identity,
                    #         "dropout": 0.1,
                    #         "batch_norm": False,
                    #     },
                    #     "pointwise_conv_config": {
                    #         "in_channels": n_coords.shape[-2],
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

                    (MEIReadIn, {
                        "meis_path": os.path.join(DATA_PATH, "meis", data_key,  "meis.pt"),
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

                    # (FCReadIn, {
                    #     "in_shape": n_coords.shape[-2],
                    #     "layers_config": [
                    #         ("fc", 432),
                    #         ("unflatten", 1, (3, 9, 16)),
                    #     ],
                    #     "act_fn": nn.LeakyReLU,
                    #     "out_act_fn": nn.Identity,
                    #     "batch_norm": True,
                    #     "dropout": 0.15,
                    # }),

                ],
            } for data_key, n_coords in _dataloaders["mouse_v1"]["train"].neuron_coords.items()
        ],
        "core_cls": GAN,
        "core_config": {
            "G_kwargs": {
                "in_shape": [480],
                "layers": [
                    ### Conv/FC readin
                    # ("deconv", 480, 7, 2, 3),
                    # ("deconv", 256, 5, 1, 2),
                    # ("deconv", 256, 5, 1, 2),
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
                "in_shape": [1, *list(crop(_dataloaders["mouse_v1"]["train"].datasets[0][0].images[0], config["crop_win"]).shape)],
                "layers": [
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
        "loss_fn": SSIMLoss(window=config["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        # "loss_fn": get_metrics(config)["SSIML-PL"],
        "l1_reg_mul": 0,
        "l2_reg_mul": 0,
        "con_reg_mul": 0,
        # "con_reg_mul": 1,
        "con_reg_loss_fn": SSIMLoss(window=config["crop_win"], log_loss=True, inp_normalized=True, inp_standardized=False),
        "encoder": None,
        # "encoder": get_encoder(
        #     ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall.pth"),
        #     device=config["device"],
        #     eval_mode=True,
        #     # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
        # ),
    },
    "val_loss": None,
    # "val_loss": get_metrics(config)["SSIML-PL"],
    "val_loss": "FID",
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
    "n_epochs": 150,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_best": False,
    #     "load_opter_state": False,
    #     "reset_history": True,
    #     "reset_best": True,
    #     # "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-25_10-16-21", "ckpt", "decoder_65.pt"),
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-18_15-23-39", "decoder.pt"),
    #     "resume_checkpointing": False,
    #     "resume_wandb_id": None,
    # },
    "save_run": True,
}
print(
    f"[INFO] List of dataloaders:"
    f"\n  TRAIN: {_dataloaders['mouse_v1']['train'].dataloaders}"
    f"\n  VAL: {_dataloaders['mouse_v1']['val'].dataloaders}"
    f"\n  TEST: {_dataloaders['mouse_v1']['test'].dataloaders}"
)
del _dataloaders


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### data
    dataloaders, neuron_coords = get_all_data(config=config)

    ### sample data
    sample_data_key = dataloaders["mouse_v1"]["val"].data_keys[0]
    datapoint = next(iter(dataloaders["mouse_v1"]["val"].dataloaders[0]))
    stim, resp, pupil_center = datapoint.images, datapoint.responses, datapoint.pupil_center

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

        # reset tracking
        if config["decoder"]["load_ckpt"]["reset_history"]:
            history = {"val_loss": []}
        if config["decoder"]["load_ckpt"]["reset_best"]:
            best = {"val_loss": np.inf, "epoch": 0, "model": None}

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
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords[sample_data_key], pupil_center=pupil_center.to(config["device"]))
        print(f"{stim_pred.shape=}")
        if stim_pred.shape != crop(stim, config["crop_win"]).shape:
            print(f"[WARNING] Stimulus prediction shape {stim_pred.shape} does not match stimulus shape {crop(stim, config['crop_win']).shape}.")
            assert stim_pred.shape[-2] >= crop(stim, config["crop_win"]).shape[-2] \
                and stim_pred.shape[-1] >= crop(stim, config["crop_win"]).shape[-1]
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

    ### wandb logging
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

    ### setup eval loss
    val_loss_fn = config["decoder"]["val_loss"]
    if val_loss_fn is None:
        val_loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])

    ### train
    print(f"[INFO] Config:\n{json.dumps(config, indent=2, default=str)}")
    s, e = len(history["val_loss"]), config["decoder"]["n_epochs"]
    for epoch in range(s, e):
        print(f"[{epoch}/{e}]")

        ### train and val
        dls, neuron_coords = get_all_data(config=config)
        train_dataloader, val_dataloader = dls["mouse_v1"]["train"], dls["mouse_v1"]["val"]
        history = train(
            model=decoder,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            config=config,
            history=history,
            epoch=epoch,
            wdb_run=wdb_run,
            wdb_commit=False,
        )
        val_losses = val(
            model=decoder,
            dataloader=val_dataloader,
            loss_fn=val_loss_fn,
            crop_win=config["crop_win"],
        )

        ### save best model
        if val_losses["total"] < best["val_loss"]:
            best["val_loss"] = val_losses["total"]
            best["epoch"] = epoch
            best["model"] = deepcopy(decoder.state_dict())

        ### log
        history["val_loss"].append(val_losses["total"])
        print(f"{val_losses['total']=:.4f}", end="")
        if config["wandb"]: wdb_run.log({"val_loss": val_losses["total"]}, commit=False)
        for data_key, loss in val_losses.items():
            if data_key != "total":
                print(f", {data_key}: {loss:.4f}", end="")
        print("")

        ### plot reconstructions
        stim_pred = decoder(
            resp[:8].to(config["device"]),
            data_key=sample_data_key,
            neuron_coords=neuron_coords[sample_data_key],
            pupil_center=pupil_center[:8].to(config["device"]),
        ).detach()
        fig = plot_comparison(target=crop(stim[:8], config["crop_win"]).cpu(), pred=crop(stim_pred[:8], config["crop_win"]).cpu(), show=False, save_to=make_sample_path(epoch, ""))
        if config["wandb"]: wdb_run.log({"val_stim_reconstruction": fig})

        ### plot losses
        if epoch % 5 == 0 and epoch > 0:
            plot_losses(history=history, epoch=epoch, show=False, save_to=os.path.join(config["dir"], f"losses_{epoch}.png") if config["decoder"]["save_run"] else None)

        ### save ckpt
        if epoch % 3 == 0 and epoch > 0 and config["decoder"]["save_run"]:
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
    dls, neuron_coords = get_all_data(config=config)
    test_loss_curr = val(
        model=decoder,
        dataloader=dls["mouse_v1"]["test"],
        loss_fn=val_loss_fn,
        crop_win=config["crop_win"],
    )
    print(f"  Test loss (current model): {test_loss_curr['total']:.4f}")

    ### load best model
    decoder.core.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "G" in k or "D" in k})
    decoder.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in best["model"].items() if "readin" in k})

    ### eval on test set w/ best params
    print("Evaluating on test set with best model...")
    dls, neuron_coords = get_all_data(config=config)
    test_loss_final = val(
        model=decoder,
        dataloader=dls["mouse_v1"]["test"],
        loss_fn=val_loss_fn,
        crop_win=config["crop_win"],
    )
    print(f"  Test loss (best model): {test_loss_final['total']:.4f}")

    ### plot reconstructions of the final model
    stim_pred_best = decoder(
        resp.to(config["device"]),
        data_key=sample_data_key,
        neuron_coords=neuron_coords[sample_data_key],
        pupil_center=pupil_center.to(config["device"]),
    ).detach().cpu()
    fig = plot_comparison(
        target=crop(stim[:8], config["crop_win"]).cpu(),
        pred=crop(stim_pred_best[:8], config["crop_win"]).cpu(),
        show=False,
        save_to=os.path.join(config["dir"], "stim_comparison_best.png") if config["decoder"]["save_run"] else None,
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

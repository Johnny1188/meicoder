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
from nnfabrik.builder import get_data

import csng
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import crop, plot_comparison, standardize, normalize, get_mean_and_std, count_parameters, plot_losses
from csng.losses import SSIMLoss, MultiSSIMLoss, Loss, CroppedLoss
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

from encoder import get_encoder
from data_utils import (
    get_mouse_v1_data,
    append_syn_dataloaders,
    append_data_aug_dataloaders,
    RespGaussianNoise,
)
from cnn_decoder_utils import train, val, get_all_data

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
    # "crop_win": (slice(7, 29), slice(15, 51)),
    "crop_win": (22, 36),
    # "wandb": None,
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
            os.path.join(DATA_PATH, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
            os.path.join(DATA_PATH, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
            os.path.join(DATA_PATH, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
            os.path.join(DATA_PATH, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
        ],
        "normalize": True,
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 7,
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
                        # "gauss_blur_sigma": "per_neuron", # "fixed", "single", "per_neuron"
                        "gauss_blur_sigma_init": 1.5,
                        "neuron_emb_dim": None,
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
                    #     "l2_reg_mul": 5e-5,
                    # }),

                    # (MEIReadIn, {
                    #     "meis_path": os.path.join(DATA_PATH, "meis", data_key,  "meis.pt"),
                    #     "n_neurons": n_coords.shape[-2],
                    #     "mei_resize_method": "resize",
                    #     "mei_target_shape": (22, 36),
                    #     "pointwise_conv_config": {
                    #         "out_channels": 480,
                    #         "bias": False,
                    #         "batch_norm": True,
                    #         "act_fn": nn.LeakyReLU,
                    #         "dropout": 0.1,
                    #     },
                    #     "ctx_net_config": {
                    #         "in_channels": 3, # resp, x, y
                    #         "layers_config": [("fc", 32), ("fc", 128), ("fc", 22*36)],
                    #         "act_fn": nn.LeakyReLU,
                    #         "out_act_fn": nn.Identity,
                    #         "dropout": 0.1,
                    #         "batch_norm": True,
                    #     },
                    #     "shift_coords": False,
                    #     "device": config["device"],
                    # }),
                    
                ],
            } for data_key, n_coords in _dataloaders["mouse_v1"]["train"].neuron_coords.items()
        ],
        "core_cls": CNN_Decoder,
        "core_config": {
            "resp_shape": (480,),
            "layers": [
                ("deconv", 480, 7, 2, 3),
                # ("conv", 480, 7, 1, 3), # MEIReadIn
                # ("deconv", 256, 7, 2, 2),
                # ("deconv", 128, 7, 2, 1),
                # ("deconv", 64, 5, 2, 2),

                ("deconv", 256, 5, 1, 2),
                # ("conv", 256, 5, 1, 2), # MEIReadIn
                # ("deconv", 128, 5, 1, 2),
                # ("deconv", 64, 5, 1, 1),

                ("deconv", 256, 5, 1, 2),
                # ("conv", 256, 5, 1, 2), # MEIReadIn
                # ("deconv", 64, 5, 1, 2),
                # ("deconv", 32, 4, 1, 1),

                ("deconv", 128, 4, 1, 1),
                # ("conv", 128, 3, 1, 1), # MEIReadIn
                # ("deconv", 64, 4, 1, 1),
                # ("deconv", 32, 4, 1, 1),

                ("deconv", 64, 3, 1, 1),
                # ("conv", 64, 3, 1, 1), # MEIReadIn
                # ("deconv", 32, 3, 1, 1),

                ("deconv", 1, 3, 1, 0),
                # ("conv", 1, 3, 1, 1), # MEIReadIn
            ],
            "act_fn": nn.ReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.35,
            "batch_norm": True,
        },
    },
    "opter_cls": torch.optim.AdamW,
    "opter_kwargs": {
        "lr": 3e-4,
        "weight_decay": 0.03,
    },
    "loss": {
        # "loss_fn": CroppedLoss(window=config["crop_win"], loss_fn=nn.MSELoss(), normalize=False, standardize=False),
        # "loss_fn": MultiSSIMLoss(
        "loss_fn": SSIMLoss(
            window=config["crop_win"],
            log_loss=True,
            inp_normalized=True,
            inp_standardized=False,
        ),
        "l1_reg_mul": 0,
        "l2_reg_mul": 0, # 1e-5
        "con_reg_mul": 0,
        # "con_reg_mul": 1,
        "con_reg_loss_fn": SSIMLoss(
            window=config["crop_win"],
            log_loss=True,
            inp_normalized=True,
            inp_standardized=False,
        ),
        "encoder": None,
        # "encoder": get_encoder(
        #     device=config["device"],
        #     eval_mode=True,
        #     # use_shifter=False,
        #     # ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_no_shifter.pth"),
        # ),
    },
    "n_epochs": 100,
    "load_ckpt": None,
    # "load_ckpt": {
    #     "load_only_core": False,
    #     # "load_only_core": True,
    #     "ckpt_path": os.path.join(
    #         # DATA_PATH, "models", "cat_v1_pretraining", "2024-02-27_19-17-39", "decoder.pt"),
    #         DATA_PATH, "models", "cnn", "2024-04-11_10-18-14", "ckpt", "decoder_45.pt"),
    #         # DATA_PATH, "models", "cnn", "2024-03-27_10-39-16", "decoder.pt"),
    #     "resume_checkpointing": True,
    #     "resume_wandb_id": "yttj8puu"
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
    sample_data_key = dataloaders["mouse_v1"]["test"].data_keys[0]
    datapoint = next(iter(dataloaders["mouse_v1"]["test"].dataloaders[0]))
    stim, resp, pupil_center = datapoint.images, datapoint.responses, datapoint.pupil_center

    ### decoder
    ### initialize (and load ckpt if needed)
    if config["decoder"]["load_ckpt"] != None:
        print(f"[INFO] Loading checkpoint from {config['decoder']['load_ckpt']['ckpt_path']}...")
        ckpt = torch.load(config["decoder"]["load_ckpt"]["ckpt_path"], map_location=config["device"], pickle_module=dill)

        if config["decoder"]["load_ckpt"]["load_only_core"]:
            print("[INFO] Loading only the core of the model (no history, no best ckpt)...")

            ### init decoder (load only the core)
            config["decoder"]["model"]["core_cls"] = ckpt["config"]["decoder"]["model"]["core_cls"]
            config["decoder"]["model"]["core_config"] = ckpt["config"]["decoder"]["model"]["core_config"]
            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict({k:v for k,v in ckpt["best"]["model"].items() if "readin" not in k}, strict=False)

            ### init the rest
            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
            history = {"train_loss": [], "val_loss": []}
            best = {"val_loss": np.inf, "epoch": 0, "model": None}
        else:
            print("[INFO] Continuing the training run (loading the current model, history, and overwriting the config)...")
            history, best, config["decoder"]["model"] = ckpt["history"], ckpt["best"], ckpt["config"]["decoder"]["model"]

            decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
            decoder.load_state_dict(ckpt["decoder"])

            opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
            opter.load_state_dict(ckpt["opter"])
            loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
    else:
        print("[INFO] Initializing the model from scratch...")
        decoder = MultiReadIn(**config["decoder"]["model"]).to(config["device"])
        opter = config["decoder"]["opter_cls"](decoder.parameters(), **config["decoder"]["opter_kwargs"])
        loss_fn = Loss(model=decoder, config=config["decoder"]["loss"])
        
        history = {"train_loss": [], "val_loss": []}
        best = {"val_loss": np.inf, "epoch": 0, "model": None}

    ### print model and fix sizes of stimuli
    with torch.no_grad():
        stim_pred = decoder(resp.to(config["device"]), data_key=sample_data_key, neuron_coords=neuron_coords[sample_data_key], pupil_center=pupil_center.to(config["device"]))
        if stim_pred.shape != crop(stim, config["crop_win"]).shape:
            print(f"[WARNING] Stimulus prediction shape {stim_pred.shape} does not match stimulus shape {crop(stim, config['crop_win']).shape}.")
            assert stim_pred.shape[-2] >= crop(stim, config["crop_win"]).shape[-2] \
                and stim_pred.shape[-1] >= crop(stim, config["crop_win"]).shape[-1]
        print(stim_pred.shape)
        del stim_pred

    print(
        decoder,
        f"\n\n---"
        f"Number of parameters:"
        f"\n  whole model: {count_parameters(decoder)}"
        f"\n  core: {count_parameters(decoder.core)} ({count_parameters(decoder.core) / count_parameters(decoder) * 100:.2f}%)"
        f"\n  readins: {count_parameters(decoder.readins)} ({count_parameters(decoder.readins) / count_parameters(decoder) * 100:.2f}%)"
        f"\n    ({', '.join([f'{k}: {count_parameters(v)} [{count_parameters(v) / count_parameters(decoder) * 100:.2f}%]' for k, v in decoder.readins.items()])})"
    )

    ### prepare checkpointing and wandb logging
    if config["decoder"]["load_ckpt"] == None \
        or config["decoder"]["load_ckpt"]["resume_checkpointing"] is False:
        config["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["decoder"]["save_run"]:
            ### save config
            config["dir"] = os.path.join(DATA_PATH, "models", "cnn", config["run_name"])
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
    else:
        wdb_run = wandb.init(**config["wandb"], name=config["run_name"], config=config, id=config["decoder"]["load_ckpt"]["resume_wandb_id"], resume="must")

    ### train
    print(f"[INFO] Config:\n{json.dumps(config, indent=2, default=str)}")
    s, e = len(history["train_loss"]), config["decoder"]["n_epochs"]
    for epoch in range(s, e):
        print(f"[{epoch}/{e}]")

        ### train and val
        dls, neuron_coords = get_all_data(config=config)
        train_dataloader, val_dataloader = dls["mouse_v1"]["train"], dls["mouse_v1"]["val"]
        train_loss = train(
            model=decoder,
            dataloader=train_dataloader,
            opter=opter,
            loss_fn=loss_fn,
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
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_losses["total"])
        if config["wandb"]: wdb_run.log({"train_loss": train_loss, "val_loss": val_losses["total"]}, commit=False)
        print(f"{train_loss=:.4f}, {val_losses['total']=:.4f}", end="")
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
        if epoch % 5 == 0 and epoch > 0:
            ### ckpt
            if config["decoder"]["save_run"]:
                torch.save({
                    "decoder": decoder.state_dict(),
                    "opter": opter.state_dict(),
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
            "opter": opter.state_dict(),
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
        loss_fn=loss_fn,
    )
    print(f"  Test loss (current model): {test_loss_curr['total']:.4f}")

    ### load best model
    decoder.load_state_dict(best["model"])

    ### eval on test set w/ best params
    print("Evaluating on test set with best model...")
    dls, neuron_coords = get_all_data(config=config)
    test_loss_final = val(
        model=decoder,
        dataloader=dls["mouse_v1"]["test"],
        loss_fn=loss_fn,
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

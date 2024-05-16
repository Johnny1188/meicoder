import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
from torch import nn
import torch.nn.functional as F
import lovely_tensors as lt

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.utils import crop, plot_comparison, standardize, normalize
from csng.comparison import load_decoder_from_ckpt, get_metrics, plot_reconstructions, plot_metrics


from encoder import get_encoder
from comparison_utils import (
    # load_decoder_from_ckpt,
    # get_metrics,
    find_best_ckpt,
    get_dataloaders,
    eval_decoder,
    # plot_reconstructions,
    # plot_metrics,
)

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
print(f"{DATA_PATH=}")


### global config
config = {
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
    },
    "crop_win": (20, 20),
    "only_cat_v1_eval": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "wandb": None,
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
    "resp_normalize_mean": None, # for evaluating Inverted Encoder
    "resp_normalize_std": None, # for evaluating Inverted Encoder
    # "resp_normalize_mean": torch.load(
    #     os.path.join(DATA_PATH, "responses_mean.pt")
    # ),
    # "resp_normalize_std": torch.load(
    #     os.path.join(DATA_PATH, "responses_std.pt")
    # ),
    # "clamp_neg_resp": True,
    "clamp_neg_resp": False, # for evaluating Inverted Encoder
}

### comparison config
config["comparison"] = {
    # "load_best": True,
    "load_best": False,
    # "eval_all_ckpts": False,
    "eval_all_ckpts": True,
    # "find_best_ckpt_according_to": None,
    "find_best_ckpt_according_to": "SSIML-PL",
    # "save_dir": None,
    "save_dir": os.path.join(
        "results",
        "table_01_new",
    ),
    "load_ckpt": None,
    "load_ckpt": {
        "overwrite": True,
        "path": os.path.join(
            "results",
            "table_01",
            "2024-04-29_11-08-54.pt",
        ),
        "load_only": None, # load all
        # "load_only": [
        #     "Inverted Encoder",
        #     # "CNN-FC (M-All)",
        #     # "CNN-Conv (M-All)",
        #     # "CNN-MEI (M-All)",
        #     # "GAN-Conv (M-All)",
        #     # "GAN-MEI (M-All)",
        # ],
        "remap": None,
        # "remap": {
        #     "CNN-FC w/ encoder matching": "CNN-FC w/ EM",
        #     "CNN-Conv w/ encoder matching": "CNN-Conv w/ EM",
        #     "CNN-MEI w/ encoder matching": "CNN-MEI w/ EM",
        #     "GAN-FC w/ encoder matching": "GAN-FC w/ EM",
        #     "GAN-Conv w/ encoder matching": "GAN-Conv w/ EM",
        #     "GAN-MEI w/ encoder matching": "GAN-MEI w/ EM",
        # },
    },
    "losses_to_plot": [
        # "SSIM",
        "SSIML",
        # "Log SSIM Loss",
        # "MultiSSIM Loss",
        # "Log MultiSSIM Loss",
        "MSE",
        # "MAE",
        "PL",
    ],
}

### Table 1
config["comparison"]["to_compare"] = {
    "Inverted Encoder": {
        "decoder": InvertedEncoder(
            encoder=get_encoder(
                device=config["device"],
                eval_mode=True,
                ckpt_path=os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter_mean_activity.pth"),
            ),
            img_dims=(1, 50, 50),
            stim_pred_init="zeros",
            opter_cls=torch.optim.SGD,
            opter_config={"lr": 10},
            n_steps=100,
            resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
            stim_loss_fn=None,
            img_gauss_blur_config=None,
            img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 1.5},
            device=config["device"],
        ).to(config["device"]),
        "run_name": None,
    },

    "CNN-FC": {
        "run_name": "2024-04-04_10-22-53",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-04_10-22-53", "decoder.pt"),
    },
    "CNN-FC w/ EM": {
        "run_name": "2024-04-26_22-13-19",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-26_22-13-19", "decoder.pt"),
    },
    "CNN-Conv": {
        "run_name": "2024-04-03_22-53-01",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-03_22-53-01", "decoder.pt"),
    },
    "CNN-Conv w/ EM": {
        "run_name": "2024-04-23_11-56-13",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-23_11-56-13", "decoder.pt"),
    },
    "CNN-MEI": {
        "run_name": "2024-04-20_11-09-52",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-20_11-09-52", "decoder.pt"),
    },
    "CNN-MEI w/ EM": {
        "run_name": "2024-04-25_18-06-51",
        "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-04-25_18-06-51", "decoder.pt"),
    },
    
    "GAN-FC": {
        "run_name": "2024-04-24_09-36-46",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-24_09-36-46", "decoder.pt"),
    },
    "GAN-FC w/ EM": {
        "run_name": "2024-04-26_22-18-25",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-26_22-18-25", "decoder.pt"),
    },
    "GAN-Conv": {
        "run_name": "2024-04-13_14-49-05",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-13_14-49-05", "decoder.pt"),
    },
    "GAN-Conv w/ EM": {
        "run_name": "2024-04-23_11-56-13",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-23_11-56-13", "decoder.pt"),
    },
    "GAN-MEI": {
        "run_name": "2024-04-20_21-54-09",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-20_21-54-09", "decoder.pt"),
    },
    "GAN-MEI w/ EM": {
        "run_name": "2024-04-25_18-00-05",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-04-25_18-00-05", "decoder.pt"),
    },
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    
    if config["comparison"]["load_best"] and config["comparison"]["eval_all_ckpts"]:
        print("[WARNING] both the eval_all_ckpts and load_best are set to True - still loading current (not the best) decoders.")
    assert config["comparison"]["eval_all_ckpts"] is True or config["comparison"]["find_best_ckpt_according_to"] is None
    assert config["comparison"]["find_best_ckpt_according_to"] is None or config["comparison"]["load_best"] is False
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### sample data
    _, _, test_dl = get_dataloaders(config=config)
    sample_data_key = "cat_v1"
    datapoint = next(iter(test_dl))
    stim, resp, neuron_coords = datapoint[sample_data_key][0].to(config["device"]), datapoint[sample_data_key][1].to(config["device"]), datapoint[sample_data_key][2].float().to(config["device"])
    stim, resp, neuron_coords = stim[:7], resp[:7], neuron_coords[:7]

    ### load ckpt
    runs_to_compare = dict()
    if config["comparison"]["load_ckpt"] is not None:
        print(f"Loading checkpoint from {config['comparison']['load_ckpt']['path']}...")
        loaded_runs = torch.load(config["comparison"]["load_ckpt"]["path"], map_location=config["device"], pickle_module=dill)["runs"]
        if config["comparison"]["load_ckpt"]["load_only"] is not None:
            ### load only selected runs
            runs_to_compare.update({run_name: loaded_runs[run_name] for run_name in config["comparison"]["load_ckpt"]["load_only"]})
        else:
            ### load all
            runs_to_compare.update(loaded_runs)
        print(f"[INFO] Loaded from ckpt: {', '.join(list(runs_to_compare.keys()))}")

        ### remap names
        remap = config["comparison"]["load_ckpt"]["remap"]
        if remap is not None:
            for in_name, out_name in remap.items():
                if in_name not in runs_to_compare:
                    continue
                runs_to_compare[out_name] = runs_to_compare[in_name]
                del runs_to_compare[in_name]
            print(f"[INFO] Remapped from ckpt to: {', '.join(list(runs_to_compare.keys()))}")

    ### merge and reorder with current to_compare config
    _runs_to_compare = dict()
    for run_name in config["comparison"]["to_compare"].keys():
        if run_name in runs_to_compare and config["comparison"]["load_ckpt"]["overwrite"]:
            _runs_to_compare[run_name] = runs_to_compare[run_name]
        else:
            _runs_to_compare[run_name] = config["comparison"]["to_compare"][run_name]
    runs_to_compare = _runs_to_compare
    metrics = get_metrics(config=config)

    ### load and compare models
    for k in runs_to_compare.keys():
        print(f"Loading {k} model from ckpt (run name: {runs_to_compare[k]['run_name']})...")
        if "test_losses" in runs_to_compare[k]:
            print(f"  Skipping...")
            continue
        run_dict = runs_to_compare[k]
        run_name = run_dict["run_name"]
        for _k in ("test_losses", "configs", "histories", "best_val_losses", "stim_pred_best", "ckpt_paths"):
            run_dict[_k] = []

        if "decoder" in run_dict and run_dict["decoder"] is not None:
            run_dict["ckpt_paths"].append(None)
        else:
            run_dict["ckpt_paths"].append(run_dict["ckpt_path"])
            if config["comparison"]["eval_all_ckpts"]:
                ### append also all other checkpoints
                ckpts_dir = os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt")
                run_dict["ckpt_paths"].extend([os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt", ckpt_name) for ckpt_name in os.listdir(ckpts_dir)])

            ### find best ckpt according to the specified metric
            if config["comparison"]["find_best_ckpt_according_to"] is not None:
                print(f"  Finding the best ckpt out of {len(run_dict['ckpt_paths'])} according to {config['comparison']['find_best_ckpt_according_to']}...")
                run_dict["ckpt_paths"] = [find_best_ckpt(config=config, ckpt_paths=run_dict["ckpt_paths"], metrics=metrics)[0]]
                print(f"    > best ckpt: {run_dict['ckpt_paths'][0]}")

        ### eval ckpts
        print(f"  Evaluating ckpts on the test set...")
        for ckpt_path in run_dict["ckpt_paths"]:
            if "decoder" in run_dict and run_dict["decoder"] is not None:
                print(f"  Using {k} model from run_dict...")
                decoder = run_dict["decoder"]
                ckpt = None
            else:
                ### load ckpt and init
                decoder, ckpt = load_decoder_from_ckpt(config=config, ckpt_path=ckpt_path)
                run_dict["configs"].append(ckpt["config"])
                run_dict["histories"].append(ckpt["history"])
                run_dict["best_val_losses"].append(ckpt["best"]["val_loss"])

            ### get sample reconstructions
            if decoder.__class__.__name__ == "InvertedEncoder":
                stim_pred_best, _, _ = decoder(resp, stim_target=None, additional_encoder_inp={"data_key": sample_data_key})
                stim_pred_best = stim_pred_best.detach().cpu()
            else:
                stim_pred_best = decoder(resp, data_key=sample_data_key, neuron_coords=neuron_coords).detach().cpu()

            ### eval
            _, _, test_dl = get_dataloaders(config=config)
            run_dict["test_losses"].append(eval_decoder(
                model=decoder,
                dataloader=test_dl,
                loss_fns=metrics,
                config=config,
            ))

            run_dict["stim_pred_best"].append(stim_pred_best.detach().cpu())

    ### save the results
    if config["comparison"]["save_dir"]:
        print(f"Saving the results to {config['comparison']['save_dir']}")
        os.makedirs(config["comparison"]["save_dir"], exist_ok=True)
        torch.save({
                "runs": runs_to_compare,
                "config": config,
            },
            os.path.join(config["comparison"]["save_dir"], f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"),
            pickle_module=dill,
        )

    ### plotting
    print(f"Plotting...")

    # plot reconstructions
    for f_type in ("png", "pdf"):
        plot_reconstructions(
            runs=runs_to_compare,
            stim=stim,
            stim_label="Target",
            crop_win=config["crop_win"],
            save_to=os.path.join(config["comparison"]["save_dir"], f"reconstructions.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )

    # plot metrics
    for f_type in ("png", "pdf"):
        plot_metrics(
            runs_to_compare=runs_to_compare,
            losses_to_plot=config["comparison"]["losses_to_plot"],
            bar_width=0.7,
            save_to=os.path.join(config["comparison"]["save_dir"], f"metrics.{f_type}") \
                if config["comparison"]["save_dir"] else None,
        )

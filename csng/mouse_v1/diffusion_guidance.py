import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import lovely_tensors as lt
import dill
import itertools
from functools import partial
from egg.diffusion import EGG

import csng
from csng.InvertedEncoder import InvertedEncoder
from csng.utils import crop, plot_comparison, dict_to_str, standardize, normalize, count_parameters
from csng.comparison import get_metrics, load_decoder_from_ckpt
from csng.mouse_v1.encoder import get_encoder
from csng.mouse_v1.data_utils import get_mouse_v1_data
from csng.mouse_v1.comparison_utils import eval_decoder
from csng.mouse_v1.diffusion_guidance_utils import GaussianBlur, do_run, energy_fn, plot_diffusion

lt.monkey_patch()
DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


### prepare config
config = {
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "crop_win": (22, 36),
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

### Encoder inversion config
config["egg"] = {
    "encoder_path": os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
    "decoder_paths": [
        os.path.join(DATA_PATH, "models", "gan", "2024-05-19_22-13-01", "ckpt/decoder_141.pt"),
        # os.path.join(DATA_PATH, "models", "gan", "2024-04-12_11-19-16", "ckpt/decoder_40.pt"),
        # os.path.join(DATA_PATH, "models", "cnn", "2024-05-21_22-22-25", "decoder.pt"),
        # os.path.join(DATA_PATH, "models", "cnn", "2024-05-24_14-28-56", "decoder.pt"),
    ],
    "model": {
        "num_steps": 1000,
        "diffusion_artefact": "/home/sobotj11/energy-guided-diffusion/models/256x256_diffusion_uncond.pt",
    },
    "energy_scale": 2,
    "em_weight": 1,
    "dm_weight": 1,
    "dm_loss_fn": "MSE-no-standardization",
    
    "find_best_according_to": "SSIML-PL",
    "save_dir": os.path.join(DATA_PATH, "models", "diffusion_guidance"),
}

### hyperparam runs config - either manually selected or grid search
config_updates = []
config_grid_search = {
    "energy_scale": [5, 7],
    "em_weight": [1, 5, 10, 20],
    "dm_weight": [0, 0.01, 1, 5],
    "dm_loss_name": ["MSE-no-standardization"],
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["egg"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["egg"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dataloaders, neuron_coords = get_mouse_v1_data(config["data"])
    sample_data_key = dataloaders["mouse_v1"]["val"].data_keys[0]
    datapoint = next(iter(dataloaders["mouse_v1"]["val"].dataloaders[0]))
    stim, resp, pupil_center = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"]), datapoint.pupil_center.to(config["device"])

    ### get metrics
    metrics = get_metrics(crop_win=config["crop_win"], device=config["device"])
    loss_fn = metrics[config["egg"]["find_best_according_to"]]

    ### prepare config_updates
    if config_grid_search is not None:
        keys, vals = zip(*config_grid_search.items())
        config_updates.extend([dict(zip(keys, v)) for v in itertools.product(*vals)])
    print(f"[INFO] Config updates to try:\n ", "\n  ".join([dict_to_str(config_update) for config_update in config_updates]))

    ### setup models
    egg_model = EGG(**config["egg"]["model"]).to(config["device"])
    encoder = get_encoder(
        ckpt_path=config["egg"]["encoder_path"],
        device=config["device"],
        eval_mode=True,
    )
    encoder_pred = partial(encoder, data_key=sample_data_key, pupil_center=pupil_center)

    ### get reconstructions from decoders to match
    xs_zero_to_match = []
    for decoder_ckpt_path in config["egg"]["decoder_paths"]:
        decoder, _ = load_decoder_from_ckpt(
            ckpt_path=decoder_ckpt_path,
            load_best=False,
            device=config["device"],
        )
        xs_zero_to_match.append(
            crop(decoder(
                resp,
                data_key=sample_data_key,
                pupil_center=pupil_center,
                neuron_coords=neuron_coords[sample_data_key]
            ), config["crop_win"]).detach()
        )

    ### run
    best = {"config": None, "loss": np.inf, "idx": None}
    print(f"[INFO] Hyperparameter search starts.")
    for i, config_update in enumerate(config_updates):
        print(f" [{i}/{len(config_updates)}]", end="")

        ### seed
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])

        ### setup the run config
        run_config = deepcopy(config)
        run_config["egg"].update(config_update)

        ### eval on validation dataset
        energy_history, stim_pred, stim_pred_history = do_run(
            model=egg_model,
            energy_fn=partial(
                energy_fn,
                encoder_model=encoder_pred,
                target_response=resp,
                norm=xs_zero_to_match[0].norm(dim=(2,3), keepdim=True),
                em_weight=run_config["egg"]["em_weight"],
                dm_weight=run_config["egg"]["dm_weight"],
                dm_loss_fn=metrics[run_config["egg"]["dm_loss_fn"]],
                xs_zero_to_match=xs_zero_to_match,
                crop_win=config["crop_win"],
            ),
            energy_scale=run_config["egg"]["energy_scale"],
            num_timesteps=run_config["egg"]["model"]["num_steps"],
            num_samples=resp.shape[0],
            grayscale=True,
        )
        loss = loss_fn(stim_pred, stim)

        ### update best
        print(f"  loss={loss:.3f}", end="")
        if loss < best["loss"]:
            print(f" >>> new best", end="")
            best["loss"] = loss
            best["config"] = run_config
            best["idx"] = i
        print("")
        print(f"   {dict_to_str(config_update)}")

        ### save
        with open(os.path.join(run_dir, f"config_{i}_{dict_to_str(config_update, as_filename=True)}.json"), "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        torch.save({
            "run_config": run_config,
            "stim_pred": stim_pred,
            "stim_pred_history": stim_pred_history,
        }, os.path.join(run_dir, f"ckpt_{i}_{dict_to_str(config_update, as_filename=True)}.pt"), pickle_module=dill)
        plot_comparison(
            target=crop(stim[:8], config["crop_win"]).cpu(),
            pred=crop(stim_pred[:8], config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{i}_{dict_to_str(config_update, as_filename=True)}.png"),
            show=False,
        )
        plot_diffusion(
            target_image=crop(stim, config["crop_win"])[0].cpu(),
            imgs=[_stim_pred[0] for _stim_pred in stim_pred_history],
            # timesteps=(0, 10, 100, 200, 300, 400, 600, 800, 999),
            timesteps=(0, 10, 20, 30, 50, 75, 100, 150, 199),
            save_to=os.path.join(run_dir, f"decoding_history_{i}_{dict_to_str(config_update, as_filename=True)}.png"),
            show=False,
        )

    print(f"[INFO] Hyperparameter search finished. Best ({best['idx']}, loss={best['loss']}):")
    json.dumps(best["config"], indent=2, default=str)
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

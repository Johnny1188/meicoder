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
import lovely_tensors as lt
import dill
import itertools
from functools import partial

# from egg.diffusion import EGG

import csng
from csng.utils.mix import seed_all, plot_comparison, dict_to_str, slugify, check_if_data_zscored, plot_comparison, count_parameters
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import load_decoder_from_ckpt, plot_reconstructions, eval_decoder, collect_all_preds_and_targets
from csng.losses import get_metrics
from csng.data import get_dataloaders
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.mouse_v1.encoder import get_encoder as get_encoder_mouse_v1
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1
from csng.models.utils.energy_guided_diffusion import EGGDecoder, do_run, energy_fn, plot_diffusion

lt.monkey_patch()
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")

get_encoder_fns = {
    "brainreader_mouse": get_encoder_brainreader,
    "mouse_v1": get_encoder_mouse_v1,
    "cat_v1": get_encoder_cat_v1,
}
encoder_paths = {
    "brainreader_mouse": os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
    "mouse_v1": os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    "cat_v1": os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
}
encoder_input_shapes = {
    "brainreader_mouse": (36, 64),
    "mouse_v1": (36, 64),
    "cat_v1": (50, 50),
}


##### global run config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {"mixing_strategy": "sequential"},
    "crop_win": None,
    "data_name": "brainreader_mouse",
}

### data config
if config["data_name"] == "brainreader_mouse":
    config["data"]["brainreader_mouse"] = {
        "device": config["device"],
        "mixing_strategy": "sequential",
        "max_batches": None,
        "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
        "batch_size": 12,
        "sessions": [6],
        "resize_stim_to": (36, 64),
        "normalize_stim": True,
        "normalize_resp": False,
        "div_resp_by_std": True,
        "clamp_neg_resp": False,
        "additional_keys": None,
        "avg_test_resp": True,
    }
    config["crop_win"] = None
elif config["data_name"] == "cat_v1":
    config["data"]["cat_v1"] = {
        "dataset_config": {
            "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
            "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
            "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
            "image_size": [50, 50],
            "crop": False,
            "batch_size": 12,
            "stim_keys": ("stim",),
            "resp_keys": ("exc_resp", "inh_resp"),
            "return_coords": True,
            "return_ori": False,
            "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
            "cached": False,
            "stim_normalize_mean": 46.143,
            "stim_normalize_std": 24.960,
            "resp_normalize_mean": None, # don't center responses
            "resp_normalize_std": torch.load(
                os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
            ),
            "clamp_neg_resp": False,
        },
    }
    config["crop_win"] = (20, 20)
elif config["data_name"] == "mouse_v1":
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
            "z_score_responses": False,
            "scale": 0.25, # 256x144 -> 64x36
            "include_behavior": False,
            "add_behavior_as_channels": False,
            "include_eye_position": True,
            "exclude": None,
            "file_tree": True,
            "cuda": "cuda" in config["device"],
            "batch_size": 12,
            "drop_last": True,
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
    config["crop_win"] = (22, 36)


### Encoder inversion config
config["egg"] = {
    "encoder_path": encoder_paths[config["data_name"]],
    "encoder_input_shape": encoder_input_shapes[config["data_name"]],
    "decoder_paths": [
        # os.path.join(DATA_PATH, "models", "gan", "2024-05-19_22-13-01", "ckpt/decoder_141.pt"),
    ],
    "model": {
        "num_steps": 1000,
        "diffusion_artefact": os.path.join(DATA_PATH, "models", "egg", "256x256_diffusion_uncond.pt"),
    },
    "energy_scale": 2,
    "norm_constraint": 60,
    "em_weight": 1,
    "energy_freq": 1,
    
    "dm_weight": 0,
    "dm_loss_fn": "MSE",
    "approximate_xstart_for_energy": True,

    "encoder_response_as_target": False, # True -> uses GT images
    "init_reconstruction_mul_factor": None,
    "loss_fns": get_metrics(inp_zscored=check_if_data_zscored(cfg=config), crop_win=config["crop_win"], device=config["device"]),
    "save_dir": os.path.join(DATA_PATH, "models", "egg"),
    "find_best_according_to": "Alex(5) Loss",
    "max_batches": 8,
}

### hyperparam runs config - either manually selected or grid search
config_updates = [
    # dict(), # Cat V1: Alex(5) Loss=0.3892 (2025-05-03_02-27-46)
    # {"model": {"num_steps": 500, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]}},
    # {"model": {"num_steps": 750, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]}},
]
config_grid_search = None
config_grid_search = {
    "energy_scale": [1, 2, 5],
    "model": [
        {"num_steps": 1000, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]},
        {"num_steps": 500, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]},
        {"num_steps": 750, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]},
        {"num_steps": 250, "diffusion_artefact": config["egg"]["model"]["diffusion_artefact"]},
    ]
}


if __name__ == "__main__":
    print(f"... Running on {config['device']} ...")
    print(f"{DATA_PATH=}")
    print(f"{DATA_PATH_BRAINREADER=}")
    print(f"{DATA_PATH_CAT_V1=}")
    print(f"{DATA_PATH_MOUSE_V1=}")
    seed_all(config["seed"])

    ### prepares dirs
    run_dir = os.path.join(config["egg"]["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(config["egg"]["save_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to {run_dir}")

    ### get data samples for plotting
    dls = get_dataloaders(config=config)[0]["val"][config["data_name"]]
    assert len(dls.data_keys) == 1, "Only single data key supported for now."
    data_key = dls.data_keys[0]
    datapoint = next(iter(dls.dataloaders[0]))
    stim, resp = datapoint.images.to(config["device"]), datapoint.responses.to(config["device"])

    ### get metrics
    # metrics = get_metrics(inp_zscored=check_if_data_zscored(cfg=config), crop_win=config["crop_win"], device=config["device"])
    # loss_fn = metrics[config["egg"]["find_best_according_to"]]

    ### prepare config_updates
    if config_grid_search is not None:
        keys, vals = zip(*config_grid_search.items())
        config_updates.extend([dict(zip(keys, v)) for v in itertools.product(*vals)])
    print(f"[INFO] Config updates to try:\n ", "\n  ".join([dict_to_str(config_update) for config_update in config_updates]))

    ### setup models
    # egg_model = EGG(**config["egg"]["model"]).to(config["device"])
    # encoder = EnsembleInvEnc(**config["egg"]["encoder"]).to(config["device"])
    # encoder = get_encoder_fns[config["data_name"]](
    #     ckpt_path=config["egg"]["encoder_path"],
    #     eval_mode=True,
    #     device=config["device"],
    # )
    # encoder_pred = partial(encoder, data_key=data_key)
    
    ### set the target response
    # if config["egg"]["encoder_response_as_target"]:
    #     target_response = encoder_pred(stim)
    # else:
    #     target_response = resp

    # ### get reconstructions from decoders to match
    # xs_zero_to_match = []
    # for decoder_ckpt_path in config["egg"]["decoder_paths"]:
    #     raise NotImplementedError("Decoder inversion not implemented yet.")
    #     decoder, _ = load_decoder_from_ckpt(
    #         ckpt_path=decoder_ckpt_path,
    #         load_best=False,
    #         device=config["device"],
    #     )
    #     pred_x_zero = crop(decoder(
    #         resp,
    #         data_key=data_key,
    #         pupil_center=pupil_center,
    #         neuron_coords=neuron_coords[data_key]
    #     ), config["crop_win"]).detach()
    #     xs_zero_to_match.append(pred_x_zero)
    #     print(f"[INFO] loss of the x_zero to match: {loss_fn(pred_x_zero, crop(stim, config['crop_win'])):.3f}   ({decoder_ckpt_path})")

    ### run
    best = {"config": None, "loss": np.inf, "idx": None, "run_name": None}
    print(f"[INFO] Hyperparameter search starts.")
    for i, config_update in enumerate(config_updates):
        print(f" [{i}/{len(config_updates)}]", end="")

        ### seed
        seed_all(config["seed"])

        ### setup the run config
        run_config = deepcopy(config)
        run_config["egg"].update(config_update)
        run_name = f"{i}__{slugify(config_update)}"

        ### prepare initital reconstructions
        # init_imgs = None
        # if run_config["egg"]["init_reconstruction_mul_factor"] is not None \
        #     and run_config["egg"]["init_reconstruction_mul_factor"] > 0:
        #     assert len(xs_zero_to_match) > 0, "No decoder reconstructions to match."
        #     init_imgs = torch.zeros((target_response.shape[0], 3, 36, 64), device=config["device"])
        #     h_s = (init_imgs.shape[-2] - config["crop_win"][0])//2
        #     h_e = (init_imgs.shape[-2] + config["crop_win"][0])//2
        #     w_s = (init_imgs.shape[-1] - config["crop_win"][1])//2
        #     w_e = (init_imgs.shape[-1] + config["crop_win"][1])//2
        #     init_imgs[:,:, h_s:h_e, w_s:w_e] = xs_zero_to_match[0].clone()
        #     init_imgs = F.interpolate(init_imgs, size=(256, 256), mode="bilinear", align_corners=False)
        #     init_imgs = run_config["egg"]["init_reconstruction_mul_factor"] * init_imgs \
        #                 + (1 - run_config["egg"]["init_reconstruction_mul_factor"]) * torch.randn_like(init_imgs)
        #     # init_imgs = normalize(standardize(init_imgs))
        #     init_imgs = init_imgs.requires_grad_()

        ### initialize decoder
        decoder = EGGDecoder(
            encoder=get_encoder_fns[config["data_name"]](
                ckpt_path=config["egg"]["encoder_path"],
                eval_mode=True,
                device=config["device"],
            ),
            encoder_input_shape=config["egg"]["encoder_input_shape"],
            egg_model_cfg=run_config["egg"]["model"],
            crop_win=config["crop_win"],
            energy_scale=run_config["egg"]["energy_scale"],
            energy_constraint=run_config["egg"]["norm_constraint"],
            num_steps=run_config["egg"]["model"]["num_steps"],
            energy_freq=run_config["egg"]["energy_freq"],
            device=config["device"],
        ).to(config["device"])

        ### eval on validation dataset
        dls = get_dataloaders(config=config)[0]["val"]
        val_loss = eval_decoder(
            model=decoder,
            dataloaders=dls,
            loss_fns={data_key: config["egg"]["loss_fns"]},
            crop_wins={data_key: config["crop_win"]},
            max_batches=config["egg"]["max_batches"],
            eval_every_n_samples=None,
        )[data_key][config["egg"]["find_best_according_to"]]
        # energy_history, stim_pred, stim_pred_history = do_run(
        #     model=egg_model,
        #     energy_fn=partial(
        #         energy_fn,
        #         encoder_model=encoder_pred,
        #         target_response=target_response,
        #         norm=run_config["egg"]["norm_constraint"],
        #         em_weight=run_config["egg"]["em_weight"],
        #         dm_weight=run_config["egg"]["dm_weight"],
        #         dm_loss_fn=metrics[run_config["egg"]["dm_loss_fn"]],
        #         xs_zero_to_match=xs_zero_to_match,
        #         crop_win=config["crop_win"],
        #         energy_freq=run_config["egg"]["energy_freq"],
        #     ),
        #     energy_scale=run_config["egg"]["energy_scale"],
        #     num_timesteps=run_config["egg"]["model"]["num_steps"],
        #     num_samples=target_response.shape[0],
        #     grayscale=True,
        #     init_imgs=init_imgs,
        #     approximate_xstart_for_energy=run_config["egg"]["approximate_xstart_for_energy"],
        # )
        # loss = loss_fn(stim_pred, stim)

        ### update best
        print(f"  loss={val_loss:.3f}", end="")
        if val_loss < best["loss"]:
            print(f" >>> new best", end="")
            best["loss"] = val_loss
            best["config"] = run_config
            best["idx"] = i
            best["run_name"] = run_name
        print("")
        print(f"   {slugify(config_update)}")

        ### save
        with open(os.path.join(run_dir, f"config_{run_name}.json"), "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        stim_pred = decoder(resp, data_key=data_key).detach().cpu()
        torch.save({
            "run_config": run_config,
            "stim_pred": stim_pred,
            "stim_pred_history": decoder.stim_pred_history,
            "energy_history": decoder.energy_history,
        }, os.path.join(run_dir, f"ckpt_{run_name}.pt"), pickle_module=dill)
        plot_comparison(
            target=crop(stim[:8], config["crop_win"]).cpu(),
            pred=crop(stim_pred[:8], config["crop_win"]).cpu(),
            save_to=os.path.join(run_dir, f"stim_pred_{run_name}.png"),
            show=False,
        )
        plot_diffusion(
            target_image=crop(stim, config["crop_win"])[0].cpu(),
            imgs=[_stim_pred[0] for _stim_pred in decoder.stim_pred_history],
            timesteps=(0, 10, 100, 200, 300, 400, 600, 800, 999) if decoder.num_steps == 1000 else np.linspace(0, decoder.num_steps-1, 9).astype(int),
            crop_win=config["crop_win"],
            save_to=os.path.join(run_dir, f"decoding_history_{run_name}.png"),
            show=False,
        )

    print(
        f"[INFO] Hyperparameter search finished.\n"
        f"  Best ({best['idx']}, val_loss={best['loss']}):\n"
        f"  Full config: {json.dumps(best['config'], indent=2, default=str)}"
        f"  Run name: {best['run_name']}\n"
        f"  Run dir: {run_dir}"
    )
    with open(os.path.join(run_dir, f"best_config.json"), "w") as f:
        json.dump(best["config"], f, indent=4, default=str)

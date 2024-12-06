import os
import numpy as np
from datetime import datetime
import dill
import torch
from torch import nn
import torch.nn.functional as F
import lovely_tensors as lt
lt.monkey_patch()

import csng
from csng.models.inverted_encoder import InvertedEncoder, InvertedEncoderBrainreader
from csng.models.ensemble import EnsembleInvEnc
from csng.utils.mix import seed_all
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import find_best_ckpt, load_decoder_from_ckpt, plot_reconstructions, plot_metrics, eval_decoder
from csng.losses import get_metrics
from csng.data import get_dataloaders, get_sample_data
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader

### set paths
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")



### global config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
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
    # "batch_size": 1,
    "batch_size": 16,
    # "sessions": list(range(1, 3)),
    "sessions": [6],
    "resize_stim_to": (36, 64),
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}
# add crop_wins for brainreader mouse data
_dls, _ = get_dataloaders(config=config)
for data_key, dset in zip(_dls["train"]["brainreader_mouse"].data_keys, _dls["train"]["brainreader_mouse"].datasets):
    config["crop_wins"][data_key] = tuple(dset[0].images.shape[-2:])

### cat v1 data
# config["data"]["cat_v1"] = {
#     "crop_win": (20, 20),
#     "dataset_config": {
#         "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
#         "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
#         "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
#         "image_size": [50, 50],
#         "crop": False,
#         "batch_size": 6,
#         "stim_keys": ("stim",),
#         "resp_keys": ("exc_resp", "inh_resp"),
#         "return_coords": True,
#         "return_ori": False,
#         "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
#         "cached": False,
#         "stim_normalize_mean": 46.143,
#         "stim_normalize_std": 20.420,
#         "resp_normalize_mean": torch.load(
#             os.path.join(DATA_PATH_CAT_V1, "responses_mean.pt")
#         ),
#         "resp_normalize_std": torch.load(
#             os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
#         ),
#         # "training_sample_idxs": np.random.choice(45000, size=22330, replace=False),
#     },
# }
# # add crop_wins for cat v1 data
# config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]

### mouse v1 data
# config["data"]["mouse_v1"] = {
#     "dataset_fn": "sensorium.datasets.static_loaders",
#     "dataset_config": {
#         "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
#             # os.path.join(DATA_PATH_MOUSE_V1, "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
#             # os.path.join(DATA_PATH_MOUSE_V1, "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
#             os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
#             # os.path.join(DATA_PATH_MOUSE_V1, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
#         ],
#         "normalize": True,
#         "scale": 0.25, # 256x144 -> 64x36
#         "include_behavior": False,
#         "add_behavior_as_channels": False,
#         "include_eye_position": True,
#         "exclude": None,
#         "file_tree": True,
#         "cuda": "cuda" in config["device"],
#         "batch_size": 6,
#         "seed": config["seed"],
#         "use_cache": False,
#     },
#     "crop_win": (22, 36),
#     "skip_train": False,
#     "skip_val": False,
#     "skip_test": False,
#     "normalize_neuron_coords": True,
#     "average_test_multitrial": True,
#     "save_test_multitrial": True,
#     "test_batch_size": 7,
#     "device": config["device"],
# }
# # add crop_wins for mouse v1 data
# for data_key, n_coords in get_dataloaders(config=config)[0]["train"]["mouse_v1"].neuron_coords.items():
#     config["crop_wins"][data_key] = config["data"]["mouse_v1"]["crop_win"]


### comparison config
config["comparison"] = {
    "load_best": True,
    # "load_best": False,
    "eval_all_ckpts": False,
    # "eval_all_ckpts": True,
    "find_best_ckpt_according_to": None,
    # "find_best_ckpt_according_to": "FID",
    "save_dir": None,
    "save_dir": os.path.join(
        "results",
        "test",
    ),
    "load_ckpt": None,
    # "load_ckpt": {
    #     "overwrite": True,
    #     "path": os.path.join(
    #         # "results",
    #         # "test",
    #         "/home/sobotka/cs-433-project/csng/results/test/2024-12-06_20-18-50.pt",
    #     ),
    #     "load_only": None, # 'None' to load all
    #     # "load_only": [
    #     #     "Inverted Encoder",
    #     #     # "CNN-MEI (M-All)",
    #     #     # "GAN-Conv (M-All)",
    #     # ],
    #     "remap": None,
    #     # "remap": {
    #     #     "CNN-Conv w/ encoder matching": "CNN-Conv w/ EM",
    #     # },
    # },
    "losses_to_plot": [
        "SSIML",
        "MSE",
        "PL",
        "FID",
    ],
}

### methods to compare
config["comparison"]["to_compare"] = {
    # "Inverted Encoder": {
    #     "decoder": InvertedEncoder(
    #         encoder=get_encoder(
    #             device=config["device"],
    #             eval_mode=True,
    #             ckpt_path=os.path.join(DATA_PATH, "models", "encoder_mall.pth"),
    #         ),
    #         img_dims=(1, 36, 64),
    #         stim_pred_init="zeros",
    #         opter_cls=torch.optim.SGD,
    #         opter_config={"lr": 10},
    #         n_steps=1000,
    #         resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
    #         stim_loss_fn=None,
    #         img_gauss_blur_config=None,
    #         img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 1.5},
    #         device=config["device"],
    #     ).to(config["device"]),
    #     "run_name": None,
    # },
    "Inverted Encoder (B-All)": {
        "decoder": InvertedEncoderBrainreader(
            encoder=get_encoder_brainreader(
                ckpt_path=os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
                device=config["device"],
                eval_mode=True,
            ),
            img_dims=(1, 36, 64),
            stim_pred_init="randn",
            lr=1000,
            n_steps=1000,
            img_grad_gauss_blur_sigma=1.5,
            jitter=None,
            mse_reduction="per_sample_mean_sum",
            device=config["device"],
        ).to(config["device"]),
        "run_name": None,
    },
    # "Inverted Encoder (Ensemble, M-6)": {
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #         os.path.join(DATA_PATH, "models", "encoder_m6_seed0.pth"),
    #         os.path.join(DATA_PATH, "models", "encoder_m6_seed1.pth"),
    #         os.path.join(DATA_PATH, "models", "encoder_m6_seed2.pth"),
    #         os.path.join(DATA_PATH, "models", "encoder_m6_seed3.pth"),
    #         os.path.join(DATA_PATH, "models", "encoder_m6_seed4.pth"),
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 36, 64),
    #             "stim_pred_init": "randn",
    #             "lr": 3000,
    #             "n_steps": 1000,
    #             "img_grad_gauss_blur_sigma": 2.,
    #             "jitter": 0,
    #             "mse_reduction": "per_sample_mean_sum",
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },

    # "CNN-Conv (M-All)": {
    #     "run_name": "2024-08-18_00-53-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-08-18_00-53-54", "ckpt", "decoder_194.pt"),
    # },

    "GAN": {
        "run_name": "2024-11-25_19-22-15",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-25_19-22-15", "decoder.pt"),
    },
}



### main comparison pipeline
def run_comparison(cfg):
    print(f"... Running on {cfg['device']} ...")
    print(f"{DATA_PATH=}")
    seed_all(cfg["seed"])

    ### check config
    if cfg["comparison"]["load_best"] and cfg["comparison"]["eval_all_ckpts"]:
        print("[WARNING] both the eval_all_ckpts and load_best are set to True - still loading current (not the best) decoders.")
    assert cfg["comparison"]["eval_all_ckpts"] is True or cfg["comparison"]["find_best_ckpt_according_to"] is None
    assert cfg["comparison"]["find_best_ckpt_according_to"] is None or cfg["comparison"]["load_best"] is False

    ### get data samples for plotting
    dls, neuron_coords = get_dataloaders(config=cfg)
    s = get_sample_data(dls=dls, config=cfg, sample_from_tier="test")
    stim, resp, sample_dataset, sample_data_key = s["stim"].to(cfg["device"]), s["resp"].to(cfg["device"]), s["sample_dataset"], s["sample_data_key"]

    ### load previous comparison results
    runs_to_compare = dict()
    if cfg["comparison"]["load_ckpt"] is not None:
        print(f"Loading checkpoint from {cfg['comparison']['load_ckpt']['path']}...")
        loaded_runs = torch.load(cfg["comparison"]["load_ckpt"]["path"], map_location=cfg["device"], pickle_module=dill)["runs"]

        ### filter loaded runs
        if cfg["comparison"]["load_ckpt"]["load_only"] is not None:
            runs_to_compare.update({run_name: loaded_runs[run_name] for run_name in cfg["comparison"]["load_ckpt"]["load_only"]})
        else: # load all
            runs_to_compare.update(loaded_runs)
        print(f"[INFO] Loaded from ckpt: {', '.join(list(runs_to_compare.keys()))}")

        ### remap names
        remap = cfg["comparison"]["load_ckpt"]["remap"]
        if remap is not None:
            for in_name, out_name in remap.items():
                if in_name not in runs_to_compare:
                    continue
                runs_to_compare[out_name] = runs_to_compare[in_name]
                del runs_to_compare[in_name]
            print(f"[INFO] Remapped from ckpt to: {', '.join(list(runs_to_compare.keys()))}")

    ### merge and reorder with current to_compare cfg
    _runs_to_compare = dict()
    for run_name in cfg["comparison"]["to_compare"].keys():
        if run_name in runs_to_compare and cfg["comparison"]["load_ckpt"]["overwrite"]:
            _runs_to_compare[run_name] = runs_to_compare[run_name]
        else:
            _runs_to_compare[run_name] = cfg["comparison"]["to_compare"][run_name]
    runs_to_compare = _runs_to_compare
    metrics = {data_key: get_metrics(crop_win=cfg["crop_wins"][data_key], device=cfg["device"]) for data_key in cfg["crop_wins"].keys()}

    ### load and compare models
    for k in runs_to_compare.keys():
        print(f"Loading {k} model from ckpt (run name: {runs_to_compare[k]['run_name']})...")
        if "test_losses" in runs_to_compare[k]: # already loaded
            print(f"  Skipping...")
            continue

        run_dict = runs_to_compare[k]
        run_name = run_dict["run_name"]
        for _k in ("test_losses", "configs", "histories", "best_val_losses", "stim_pred_best", "ckpt_paths"):
            run_dict[_k] = []

        ### set ckpt paths
        if "decoder" in run_dict and run_dict["decoder"] is not None:
            run_dict["ckpt_paths"].append(None) # decoder directly in run_dict
        else:
            run_dict["ckpt_paths"].append(run_dict["ckpt_path"])

            ### append also all other checkpoints
            if cfg["comparison"]["eval_all_ckpts"]:
                ckpts_dir = os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt")
                run_dict["ckpt_paths"].extend([os.path.join(os.path.dirname(run_dict["ckpt_path"]), "ckpt", ckpt_name) for ckpt_name in os.listdir(ckpts_dir)])

            ### find best ckpt according to the specified metric
            if cfg["comparison"]["find_best_ckpt_according_to"] is not None:
                print(f"  Finding the best ckpt out of {len(run_dict['ckpt_paths'])} according to {cfg['comparison']['find_best_ckpt_according_to']}...")
                get_val_dl_fn = lambda: get_dataloaders(config=cfg)[0]["val"]
                run_dict["ckpt_paths"] = [find_best_ckpt(get_dl_fn=get_val_dl_fn, config=cfg, ckpt_paths=run_dict["ckpt_paths"], metrics=metrics)[0]]
                print(f"    > best ckpt: {run_dict['ckpt_paths'][0]}")

        ### eval ckpts
        print(f"  Evaluating checkpoints on the test set...")
        for ckpt_path in run_dict["ckpt_paths"]:
            if "decoder" in run_dict and run_dict["decoder"] is not None:
                print(f"  Using {k} model from run_dict...")
                decoder = run_dict["decoder"]
                ckpt = None
            else:
                ### load ckpt and init
                decoder, ckpt = load_decoder_from_ckpt(ckpt_path=ckpt_path, device=cfg["device"], load_best=cfg["comparison"]["load_best"], load_only_core=False, strict=True)
                run_dict["configs"].append(ckpt["config"])
                run_dict["histories"].append(ckpt["history"])
                run_dict["best_val_losses"].append(ckpt["best"]["val_loss"])

            ### get sample reconstructions
            stim_pred_best = dict()
            if "brainreader_mouse" in cfg["data"]:
                stim_pred_best[s["b_sample_data_key"]] = decoder(s["b_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach().cpu()
            if "cat_v1" in cfg["data"]:
                stim_pred_best[s["c_sample_data_key"]] = decoder(s["c_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]], data_key=s["c_sample_data_key"]).detach().cpu()
            if "mouse_v1" in cfg["data"]:
                stim_pred_best[s["m_sample_data_key"]] = decoder(s["m_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach().cpu()
            run_dict["stim_pred_best"].append(stim_pred_best)

            ### eval
            seed_all(cfg["seed"])
            dls, _ = get_dataloaders(config=cfg)
            run_dict["test_losses"].append(eval_decoder(
                model=decoder,
                dataloaders=dls["test"],
                loss_fns=metrics,
                crop_wins=cfg["crop_wins"],
                calc_fid="FID" in cfg["comparison"]["losses_to_plot"],
            ))

    ### save the results
    if cfg["comparison"]["save_dir"]:
        print(f"Saving the results to {cfg['comparison']['save_dir']}")
        os.makedirs(cfg["comparison"]["save_dir"], exist_ok=True)
        torch.save({
                "runs": runs_to_compare,
                "config": cfg,
            },
            os.path.join(cfg["comparison"]["save_dir"], f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"),
            pickle_module=dill,
        )

    ### plot reconstructions
    print(f"Plotting reconstructions...")
    for f_type in ("png", "pdf"):
        for data_key in cfg["crop_wins"].keys():
            plot_reconstructions(
                runs=runs_to_compare,
                stim=stim,
                stim_label="Target",
                data_key=data_key,
                crop_win=cfg["crop_wins"][data_key],
                save_to=os.path.join(cfg["comparison"]["save_dir"], f"reconstructions_{data_key}.{f_type}") \
                    if cfg["comparison"]["save_dir"] else None,
            )

    ### plot metrics
    print(f"Plotting metrics...")
    for f_type in ("png", "pdf"):
        plot_metrics(
            runs_to_compare=runs_to_compare,
            losses_to_plot=cfg["comparison"]["losses_to_plot"],
            bar_width=0.7,
            save_to=os.path.join(cfg["comparison"]["save_dir"], f"metrics.{f_type}") \
                if cfg["comparison"]["save_dir"] else None,
        )


if __name__ == "__main__":
    run_comparison(cfg=config)

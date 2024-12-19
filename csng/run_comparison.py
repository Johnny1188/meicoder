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
    # "batch_size": 4,
    "batch_size": 16,
    # "sessions": list(range(1, 23)),
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
### add neuron coordinates to brainreader mouse data (learned by pretrained encoder)
_enc_ckpt = torch.load(os.path.join(DATA_PATH, "models", "encoder_ball.pt"), pickle_module=dill)["model"]
config["data"]["brainreader_mouse"]["neuron_coords"] = dict()
for sess_id in config["data"]["brainreader_mouse"]["sessions"]:
    config["data"]["brainreader_mouse"]["neuron_coords"][str(sess_id)] = _enc_ckpt[f"readout.{sess_id}._mu"][0,:,0].detach().clone()


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
    "eval_all_ckpts": False,
    "find_best_ckpt_according_to": None, # "FID"
    "eval_tier": "test",
    "save_dir": None,
    "save_dir": os.path.join(
        "results",
        "inv_enc",
    ),
    "load_ckpt": None,
    # "load_ckpt": {
    #     "overwrite": True,
    #     "path": os.path.join(
    #         "results/test/2024-12-06_20-18-50.pt",
    #     ),
    #     "load_only": None, # 'None' to load all
    #     # "load_only": [
    #     #     "Inverted Encoder",
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
        "FID",
    ],
}

### methods to compare
config["comparison"]["to_compare"] = {
    ### --- Inverted encoder ---
    "Inverted Encoder": {
        "decoder": EnsembleInvEnc(
            encoder_paths=[
                os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
            ],
            encoder_config={
                "img_dims": (1, 36, 64),
                "stim_pred_init": "randn",
                "lr": 2000,
                "n_steps": 1000,
                "img_grad_gauss_blur_sigma": 1,
                "jitter": None,
                "mse_reduction": "per_sample_mean_sum",
                "device": config["device"],
            },
            use_brainreader_encoder=True,
            get_encoder_fn=get_encoder_brainreader,
            device=config["device"],
        ),
        "run_name": None,
    },


    ### --- CNN MSE ---
    # "CNN": {
    #     "run_name": "2024-12-17_03-20-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-12-17_03-20-48", "decoder.pt"),
    # },


    ### --- Hyperparameter search - number of channels (seed 0) ---
    ## Seed 0
    # "GAN (64)": {
    #     "run_name": "2024-12-14_05-11-25",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_05-11-25", "decoder.pt"),
    # },
    # "GAN (256)": {
    #     "run_name": "2024-12-09_01-13-05",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-09_01-13-05", "decoder.pt"),
    # },
    # "GAN (480)": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (624)": {
    #     "run_name": "2024-12-09_01-22-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-09_01-22-11", "decoder.pt"),
    # },
    # "GAN (864)": {
    #     "run_name": "2024-12-10_01-58-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_01-58-14", "decoder.pt"),
    # },
    # "GAN (1028)": {
    #     "run_name": "2024-12-10_02-04-31",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-04-31", "decoder.pt"),
    # },
    # "GAN (1256)": {
    #     "run_name": "2024-12-10_02-14-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-14-34", "decoder.pt"),
    # },


    ### --- Hyperparameter search - number of channels (seed 1) ---
    # "GAN (64)": {
    #     "run_name": "2024-12-15_02-24-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_02-24-54", "decoder.pt"),
    # },
    # "GAN (256)": {
    #     "run_name": "2024-12-12_09-14-47",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-14-47", "decoder.pt"),
    # },
    # "GAN (480)": {
    #     "run_name": "2024-12-12_09-13-47",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-13-47", "decoder.pt"),
    # },
    # "GAN (624)": {
    #     "run_name": "2024-12-12_09-10-30",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-10-30", "decoder.pt"),
    # },
    # "GAN (864)": {
    #     "run_name": "2024-12-12_09-10-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-10-14", "decoder.pt"),
    # },
    # "GAN (1028)": {
    #     "run_name": "2024-12-12_09-12-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-12-02", "decoder.pt"),
    # },
    # "GAN (1256)": {
    #     "run_name": "2024-12-12_09-12-56",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-12-56", "decoder.pt"),
    # },


    ### --- Hyperparameter search - number of channels (seed 2) ---
    # "GAN (64)": {
    #     "run_name": "2024-12-15_02-25-55",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_02-25-55", "decoder.pt"),
    # },
    # "GAN (256)": {
    #     "run_name": "2024-12-13_03-29-23",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-29-23", "decoder.pt"),
    # },
    # "GAN (480)": {
    #     "run_name": "2024-12-13_03-20-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-20-42", "decoder.pt"),
    # },
    # "GAN (624)": {
    #     "run_name": "2024-12-13_03-17-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-17-34", "decoder.pt"),
    # },
    # "GAN (864)": {
    #     "run_name": "2024-12-13_03-10-55",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-10-55", "decoder.pt"),
    # },
    # "GAN (1028)": {
    #     "run_name": "2024-12-13_02-49-56",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_02-49-56", "decoder.pt"),
    # },
    # "GAN (1256)": {
    #     "run_name": "2024-12-13_02-49-49",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_02-49-49", "decoder.pt"),
    # },


    ### --- Varying number of neurons ---
    # "GAN (100%)": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (87.5%)": {
    #     "run_name": "2024-12-15_03-54-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_03-54-50", "decoder.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2024-12-11_03-24-58",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-24-58", "decoder.pt"),
    # },
    # "GAN (62.5%)": {
    #     "run_name": "2024-12-15_04-23-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_04-23-33", "decoder.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2024-12-11_03-20-53",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-20-53", "decoder.pt"),
    # },
    # "GAN (37.5%)": {
    #     "run_name": "2024-12-15_03-54-43",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_03-54-43", "decoder.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2024-12-11_03-22-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-22-07", "decoder.pt"),
    # },
    # "GAN (10%)": {
    #     "run_name": "2024-12-11_03-27-17",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-27-17", "decoder.pt"),
    # },
    # "GAN (5%)": {
    #     "run_name": "2024-12-11_03-47-17",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-47-17", "decoder.pt"),
    # },


    ### --- Varying number of training data batches ---
    # "GAN (100%)": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2024-12-12_00-40-09",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_00-40-09", "decoder.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2024-12-11_03-53-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-53-59", "decoder.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2024-12-11_20-20-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_20-20-59", "decoder.pt"),
    # },
    # "GAN (10%)": {
    #     "run_name": "2024-12-11_20-42-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_20-42-02", "decoder.pt"),
    # },
    # "GAN (5%)": {
    #     "run_name": "2024-12-12_01-22-10",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_01-22-10", "decoder.pt"),
    # },
    # "GAN (2.5%)": {
    #     "run_name": "2024-12-12_01-22-26",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_01-22-26", "decoder.pt"),
    # },
    # "GAN (1%)": {
    #     "run_name": "2024-12-12_02-04-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_02-04-04", "decoder.pt"),
    # },


    ### --- B-All ---
    # "GAN (B-All)": {
    #     "run_name": "2024-11-19_15-45-08",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-19_15-45-08", "decoder.pt"),
    # },
    # r"GAN (B-All $\rightarrow$ B-6)": {
    #     "run_name": "2024-12-11_03-44-45",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-44-45", "decoder.pt"),
    # },
    # "GAN (50% B-All + 50% S-All)": {
    #     "run_name": "2024-11-28_01-03-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-28_01-03-50", "decoder.pt"),
    # },
    # r"GAN (50% B-All + 50% S-All $\rightarrow$ B-6)": {
    #     "run_name": "2024-11-28_01-03-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-28_01-03-50", "decoder.pt"),
    # },


    ### --- With(out) coordinates ---
    # "GAN": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (w/ coordinates)": {
    #     "run_name": "2024-12-08_20-08-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-08_20-08-02", "decoder.pt"),
    # },
    # "GAN (w/ coordinates & transformation)": {
    #     "run_name": "2024-12-08_21-03-52",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-08_21-03-52", "decoder.pt"),
    # },


    ### --- With(out) resp_transform ---
    # "GAN": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (w/ transformation)": {
    #     "run_name": "2024-12-10_03-29-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_03-29-36", "decoder.pt"),
    # },
    # "GAN (w/ coordinates & transformation)": {
    #     "run_name": "2024-12-08_21-03-52",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-08_21-03-52", "decoder.pt"),
    # },


    ### --- With(out) resp_transform & coordinates (seed 0) ---
    # "GAN": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (w/ C)": {
    #     "run_name": "2024-12-08_20-08-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-08_20-08-02", "decoder.pt"),
    # },
    # "GAN (w/ T)": {
    #     "run_name": "2024-12-10_03-29-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_03-29-36", "decoder.pt"),
    # },
    # "GAN (w/ C & T)": {
    #     "run_name": "2024-12-08_21-03-52",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-08_21-03-52", "decoder.pt"),
    # },


    ### --- With(out) resp_transform & coordinates (seed 1) ---
    # "GAN": {
    #     "run_name": "2024-12-12_09-13-47",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-13-47", "decoder.pt"),
    # },
    # "GAN (w/ C)": {
    #     "run_name": "2024-12-14_03-48-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_03-48-11", "decoder.pt"),
    # },
    # "GAN (w/ T)": {
    #     "run_name": "2024-12-14_03-52-10",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_03-52-10", "decoder.pt"),
    # },
    # "GAN (w/ C & T)": {
    #     "run_name": "2024-12-14_03-42-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_03-42-14", "decoder.pt"),
    # },


    ### --- With(out) resp_transform & coordinates (seed 2) ---
    # "GAN": {
    #     "run_name": "2024-12-13_03-20-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-20-42", "decoder.pt"),
    # },
    # "GAN (w/ C)": {
    #     "run_name": "2024-12-14_04-35-56",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_04-35-56", "decoder.pt"),
    # },
    # "GAN (w/ T)": {
    #     "run_name": "2024-12-14_05-01-51",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_05-01-51", "decoder.pt"),
    # },
    # "GAN (w/ C & T)": {
    #     "run_name": "2024-12-14_04-37-22",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_04-37-22", "decoder.pt"),
    # },


    # ### --- Training loss (seed 0) ---
    # "GAN (SSIML)": {
    #     "run_name": "2024-12-10_02-52-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-10_02-52-29", "decoder.pt"),
    # },
    # "GAN (MSE)": {
    #     "run_name": "2024-11-24_13-08-45",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-24_13-08-45", "decoder.pt"),
    # },
    # "GAN (MAE)": {
    #     "run_name": "2024-12-12_02-12-39",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_02-12-39", "decoder.pt"),
    # },


    ### --- Training loss (seed 1) ---
    # "GAN (SSIML)": {
    #     "run_name": "2024-12-12_09-13-47",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_09-13-47", "decoder.pt"),
    # },
    # "GAN (MSE)": {
    #     "run_name": "2024-12-14_05-16-56",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-14_05-16-56", "decoder.pt"),
    # },
    # "GAN (MAE)": {
    #     "run_name": "2024-12-13_03-20-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-12_02-12-39", "decoder.pt"),
    # },


    ### --- Training loss (seed 2) ---
    # "GAN (SSIML)": {
    #     "run_name": "2024-12-13_03-20-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_03-20-42", "decoder.pt"),
    # },
    # "GAN (MSE)": {
    #     "run_name": "2024-12-15_02-18-42",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_02-18-42", "decoder.pt"),
    # },
    # "GAN (MAE)": {
    #     "run_name": "2024-12-15_01-58-58",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-15_01-58-58", "decoder.pt"),
    # },


    ### --- Synthetic training data ---
    # "GAN (92%)": {
    #     "run_name": "2024-11-22_00-05-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-22_00-05-07", "decoder.pt"),
    # },
    # "GAN (92%, shared readin)": {
    #     "run_name": "2024-11-28_00-48-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-28_00-48-07", "decoder.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2024-11-24_02-18-51",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-24_02-18-51", "decoder.pt"),
    # },
    # "GAN (75%, shared readin)": {
    #     "run_name": "2024-12-11_03-42-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-42-04", "decoder.pt"),
    # },
    # "GAN (75%, shared readin, FT)": {
    #     "run_name": "2024-12-13_02-26-40",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-13_02-26-40", "decoder.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2024-11-22_00-35-18",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-22_00-35-18", "decoder.pt"),
    # },
    # "GAN (50%, shared readin)": {
    #     "run_name": "2024-11-28_00-38-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-28_00-38-48", "decoder.pt"),
    # },
    # "GAN (50%, shared readin, FT)": {
    #     "run_name": "2024-12-11_03-28-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-28-54", "decoder.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2024-11-24_02-12-30",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-24_02-12-30", "decoder.pt"),
    # },
    # "GAN (25%, shared readin)": {
    #     "run_name": "2024-11-30_01-36-31",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-11-30_01-36-31", "decoder.pt"),
    # },
    # "GAN (25%, shared readin, FT)": {
    #     "run_name": "2024-12-11_03-35-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2024-12-11_03-35-04", "decoder.pt"),
    # },
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
                dataloaders=dls[cfg["comparison"]["eval_tier"]],
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

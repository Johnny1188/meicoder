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
from csng.models.inverted_encoder_decoder import InvertedEncoderDecoder
from csng.models.utils.energy_guided_diffusion import EGGDecoder, do_run, energy_fn, plot_diffusion
from csng.utils.mix import seed_all, check_if_data_zscored, update_config_paths, update_config
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import find_best_ckpt, load_decoder_from_ckpt, plot_reconstructions, plot_metrics, eval_decoder, SavedReconstructionsDecoder, collect_all_preds_and_targets
from csng.losses import get_metrics
from csng.data import get_dataloaders, get_sample_data
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.mouse_v1.encoder import get_encoder as get_encoder_sensorium_mouse_v1
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1

from monkeysee.SpatialBased.decoding_wrapper import MonkeySeeDecoder
# from cae.model import CAEDecoder

### set paths
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAE = os.path.join(os.environ["DATA_PATH"], "cae")
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
# config["data"]["brainreader_mouse"] = {
#     "device": config["device"],
#     "mixing_strategy": config["data"]["mixing_strategy"],
#     "max_batches": None,
#     "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
#     # "batch_size": 4,
#     "batch_size": 12,
#     # "batch_size": 36,
#     # "sessions": list(range(1, 23)),
#     "sessions": [6],
#     "drop_last": True,
#     "resize_stim_to": (36, 64),
#     "normalize_stim": True,
#     "normalize_resp": False,
#     "div_resp_by_std": True,
#     "clamp_neg_resp": False,
#     "additional_keys": None,
#     "avg_test_resp": True,
# }
# # add crop_wins for brainreader mouse data
# _dls, _ = get_dataloaders(config=config)
# for data_key, dset in zip(_dls["train"]["brainreader_mouse"].data_keys, _dls["train"]["brainreader_mouse"].datasets):
#     config["crop_wins"][data_key] = tuple(dset[0].images.shape[-2:])
# ## add neuron coordinates to brainreader mouse data (learned by pretrained encoder)
# _enc_ckpt = torch.load(os.path.join(DATA_PATH, "models", "encoder_ball.pt"), pickle_module=dill)["model"]
# config["data"]["brainreader_mouse"]["neuron_coords"] = dict()
# for sess_id in config["data"]["brainreader_mouse"]["sessions"]:
#     config["data"]["brainreader_mouse"]["neuron_coords"][str(sess_id)] = _enc_ckpt[f"readout.{sess_id}._mu"][0,:,0].detach().clone()


### cat v1 data
config["data"]["cat_v1"] = {
    "crop_win": (20, 20),
    "dataset_config": {
        "train_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "train"),
        "val_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "val"),
        "test_path": os.path.join(DATA_PATH_CAT_V1, "datasets", "test"),
        "image_size": [50, 50],
        "crop": False,
        "batch_size": 12,
        # "batch_size": 36,
        "stim_keys": ("stim",),
        "resp_keys": ("exc_resp", "inh_resp"),
        "return_coords": True,
        "return_ori": False,
        "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
        "cached": False,
        "stim_normalize_mean": 46.143,
        "stim_normalize_std": 24.960,
        "resp_normalize_mean": None,
        "resp_normalize_std": torch.load(
            os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
        ),
    },
}
# add crop_wins for cat v1 data
config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]

### mouse v1 data
# config["data"]["mouse_v1"] = {
#     "dataset_fn": "sensorium.datasets.static_loaders",
#     "dataset_config": {
#         "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
#             os.path.join(DATA_PATH_MOUSE_V1, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-1
#             # os.path.join(DATA_PATH_MOUSE_V1, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-2
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-3
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-4
#             # os.path.join(DATA_PATH_MOUSE_V1, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # M-5
#         ],
#         "normalize": True,
#         "z_score_responses": False,
#         "scale": 0.25, # 256x144 -> 64x36
#         "include_behavior": False,
#         "add_behavior_as_channels": False,
#         "include_eye_position": True,
#         "exclude": None,
#         "file_tree": True,
#         "cuda": "cuda" in config["device"],
#         "batch_size": 16,
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
#     "test_batch_size": 12,
#     # "test_batch_size": 36,
#     "device": config["device"],
# }
# ### add crop_wins for mouse v1 data
# for data_key, n_coords in get_dataloaders(config=config)[0]["train"]["mouse_v1"].neuron_coords.items():
#     config["crop_wins"][data_key] = config["data"]["mouse_v1"]["crop_win"]


### comparison config
config["comparison"] = {
    "load_best": True,
    "eval_all_ckpts": False,
    "find_best_ckpt_according_to": None, # "Alex(5) Loss"
    "eval_tier": "test",
    "eval_every_n_samples": None, # to prevent OOM but not accurate for some losses
    "max_n_reconstruction_samples": None,
    "z_score_wrt_target": False,
    "save_all_preds_and_targets": False,
    "save_dir": None,
    "save_dir": os.path.join(
        DATA_PATH,
        "results",
        "semantic_catv1_meicoder",
    ),
    "load_ckpt": None,
    # "load_ckpt": {
    #     "overwrite": True,
    #     "path": os.path.join(
    #         "/home/jan/Desktop/Dev/csng/decoding-brain-activity/publication/figures/12-04-25/brainreader/2025-04-12_10-34-28.pt",
    #     ),
    #     "load_only": None, # 'None' to load all
    #     # "load_only": [
    #     #     "Inverted Encoder",
    #     #     # "GAN-Conv (M-All)",
    #     # ],
    #     "remap": None,
    #     # "remap": {
    #     #     "Inverted Encoder": "Inverted Encoder (brainreader-style)",
    #     # },
    # },
    "losses_to_plot": [
        "SSIM",
        "PixCorr",
        "Alex(2)",
        "Alex(5)",
        # "BrainDistance",
        "Incep",
        "CLIP",
        "Eff",
        "SwAV",
    ],
}

### methods to compare
config["comparison"]["to_compare"] = {
    ### --- CAEDecoder ---
    # "CAE": { # B-6
    #     "decoder": CAEDecoder(
    #         ckpt_path=os.path.join(DATA_PATH_CAE, "runs", "27-07-2025_15-13", "best_model.pt"),
    #     ).to(config["device"]),
    #     "run_name": None,
    # },
    # "CAE": { # M-1
    #     "decoder": CAEDecoder(
    #         ckpt_path=os.path.join(DATA_PATH_CAE, "runs", "27-07-2025_16-43", "best_model.pt"),
    #     ).to(config["device"]),
    #     "run_name": None,
    # },
    # "CAE": { # Cat V1
    #     "decoder": CAEDecoder(
    #         ckpt_path=os.path.join(DATA_PATH_CAE, "runs", "27-07-2025_17-53", "best_model.pt"),
    #     ).to(config["device"]),
    #     "run_name": None,
    # },

    ### --- Inverted encoder ---
    # "Inverted Encoder (brainreader-style)": { # brainreader mouse
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
    #             # os.path.join(DATA_PATH, "models", "encoder_b6.pt"),
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 36, 64),
    #             "stim_pred_init": "randn",
    #             # "lr": 2000,
    #             # "n_steps": 1000,
    #             "lr": 500,
    #             "n_steps": 2000,
    #             "img_grad_gauss_blur_sigma": 1.5,
    #             "jitter": None,
    #             "mse_reduction": "per_sample_mean_sum",
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=True,
    #         get_encoder_fn=get_encoder_brainreader,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "Inverted Encoder": { # brainreader mouse
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             # os.path.join(DATA_PATH, "models", "encoder_ball.pt"),
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_b6_seed1.pt"), # seed=1
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_b6_seed2.pt"), # seed=2
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 36, 64),
    #             "stim_pred_init": "zeros",
    #             "opter_config": {"lr": 50},
    #             "n_steps": 1000,
    #             "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.},
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=False,
    #         get_encoder_fn=get_encoder_brainreader,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "Inverted Encoder (brainreader-style)": { # sensorium mouse v1
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 36, 64),
    #             "stim_pred_init": "randn",
    #             "lr": 500,
    #             "n_steps": 2000,
    #             "img_grad_gauss_blur_sigma": 1,
    #             "jitter": None,
    #             "mse_reduction": "per_sample_mean_sum",
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=True,
    #         get_encoder_fn=get_encoder_sensorium_mouse_v1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "Inverted Encoder": { # sensorium mouse v1
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_m1_seed1.pt"), # seed=1
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_m1_seed2.pt"), # seed=2
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 36, 64),
    #             "stim_pred_init": "zeros",
    #             "opter_config": {"lr": 50},
    #             "n_steps": 1000,
    #             "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.},
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=False,
    #         get_encoder_fn=get_encoder_sensorium_mouse_v1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "Inverted Encoder (brainreader-style)": { # cat v1
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 50, 50),
    #             "stim_pred_init": "randn",
    #             # "lr": 500,
    #             # "n_steps": 2000,
    #             "lr": 1000,
    #             "n_steps": 2000,
    #             "img_grad_gauss_blur_sigma": 1.5,
    #             "jitter": None,
    #             "mse_reduction": "per_sample_mean_sum",
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=True,
    #         get_encoder_fn=get_encoder_cat_v1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "Inverted Encoder": { # cat v1
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_c_seed1.pt"), # seed=1
    #             # os.path.join(DATA_PATH, "models", "encoders", "encoder_c_seed2.pt"), # seed=2
    #         ],
    #         encoder_config={
    #             "img_dims": (1, 50, 50),
    #             "stim_pred_init": "zeros",
    #             "opter_config": {"lr": 10},
    #             "n_steps": 1000,
    #             "img_grad_gauss_blur_config": {"kernel_size": 13, "sigma": 1.5},
    #             "device": config["device"],
    #         },
    #         use_brainreader_encoder=False,
    #         get_encoder_fn=get_encoder_cat_v1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },

    ### --- Energy guided diffusion ---
    # "EGG": { # brainreader mouse
    #     "decoder": EGGDecoder(
    #         encoder=get_encoder_brainreader(
    #             ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
    #             eval_mode=True,
    #             device=config["device"],
    #         ),
    #         encoder_input_shape=(36, 64),
    #         egg_model_cfg={
    #             "num_steps": (egg_num_steps := 750),
    #             "diffusion_artefact": os.path.join(DATA_PATH, "models", "egg", "256x256_diffusion_uncond.pt"),
    #         },
    #         crop_win=config["crop_wins"]["6"],
    #         energy_scale=1,
    #         energy_constraint=60,
    #         num_steps=egg_num_steps,
    #         energy_freq=1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "EGG": { # mouse v1
    #     "decoder": EGGDecoder(
    #         encoder=get_encoder_sensorium_mouse_v1(
    #             ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    #             # ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_m1_seed1.pt"), # seed=1
    #             # ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_m1_seed2.pt"), # seed=2
    #             eval_mode=True,
    #             device=config["device"],
    #         ),
    #         encoder_input_shape=(36, 64),
    #         egg_model_cfg={
    #             "num_steps": (egg_num_steps := 250),
    #             "diffusion_artefact": os.path.join(DATA_PATH, "models", "egg", "256x256_diffusion_uncond.pt"),
    #         },
    #         crop_win=config["crop_wins"]["21067-10-18"],
    #         energy_scale=5,
    #         energy_constraint=60,
    #         num_steps=egg_num_steps,
    #         energy_freq=1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "EGG": { # cat v1
    #     "decoder": EGGDecoder(
    #         encoder=get_encoder_cat_v1(
    #             ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
    #             eval_mode=True,
    #             device=config["device"],
    #         ),
    #         encoder_input_shape=(50, 50),
    #         egg_model_cfg={
    #             "num_steps": (egg_num_steps := 100),
    #             "diffusion_artefact": os.path.join(DATA_PATH, "models", "egg", "256x256_diffusion_uncond.pt"),
    #         },
    #         crop_win=config["crop_wins"]["cat_v1"],
    #         energy_scale=5,
    #         energy_constraint=60,
    #         num_steps=egg_num_steps,
    #         energy_freq=1,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },

    ### --- MonkeySee ---
    # "MonkeySee": { # B-6
    #     "decoder": MonkeySeeDecoder(
    #         ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "18-02-2025_19-32")),
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "12-05-2025_04-13")), # seed=1
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "12-05-2025_15-33")), # seed=2
    #         ckpt_key_to_load="best_es",
    #         train_dl=get_dataloaders(config=(monkeysee_config := update_config(
    #                 config=update_config_paths(
    #                     config=torch.load(os.path.join(monkeysee_ckpt_path, "generator.pt"), pickle_module=dill)["config"],
    #                     new_data_path=DATA_PATH,
    #                 ),
    #                 config_updates={
    #                     "data__brainreader_mouse__batch_size": config["data"]["brainreader_mouse"]["batch_size"],
    #                     "data__brainreader_mouse__drop_last": config["data"]["brainreader_mouse"]["drop_last"],
    #                 }
    #             )
    #         ))[0]["train"]["brainreader_mouse"],
    #         new_data_path=DATA_PATH,
    #     ),
    #     "use_data_config": monkeysee_config,
    #     "run_name": None,
    # },
    # "MonkeySee": { # M-1
    #     "decoder": MonkeySeeDecoder(
    #         ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "26-03-2025_03-02")),
    #         ckpt_key_to_load="best_es",
    #         train_dl=get_dataloaders(config=(monkeysee_config := update_config(
    #                 config=update_config_paths(
    #                     config=torch.load(os.path.join(monkeysee_ckpt_path, "generator.pt"), pickle_module=dill)["config"],
    #                     new_data_path=DATA_PATH,
    #                     replace_until_folder="csng",
    #                 ),
    #                 config_updates={
    #                     "data__mouse_v1__test_batch_size": config["data"]["mouse_v1"]["test_batch_size"],
    #                 }
    #             )
    #         ))[0]["train"]["mouse_v1"],
    #         new_data_path=DATA_PATH,
    #     ),
    #     "use_data_config": monkeysee_config,
    #     "run_name": None,
    # },
    # "MonkeySee": { # M-1
    #     "decoder": MonkeySeeDecoder(
    #         ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "26-04-2025_01-46")),
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "11-05-2025_02-26")), # seed=1
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "11-05-2025_15-19")), # seed=2
    #         ckpt_key_to_load="best_es",
    #         train_dl=get_dataloaders(config=(monkeysee_config := update_config(
    #                 config=update_config_paths(
    #                     config=torch.load(os.path.join(monkeysee_ckpt_path, "generator.pt"), pickle_module=dill)["config"],
    #                     new_data_path=DATA_PATH,
    #                     replace_until_folder="csng",
    #                 ),
    #                 config_updates={
    #                     "data__mouse_v1__test_batch_size": config["data"]["mouse_v1"]["test_batch_size"],
    #                 }
    #             )
    #         ))[0]["train"]["mouse_v1"],
    #         new_data_path=DATA_PATH,
    #     ),
    #     "use_data_config": monkeysee_config,
    #     "run_name": None,
    # },
    # "MonkeySee": { # cat v1
    #     "decoder": MonkeySeeDecoder(
    #         ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "23-02-2025_11-00")),
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "11-05-2025_00-17")),
    #         # ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "11-05-2025_01-55")),
    #         ckpt_key_to_load="best_es",
    #         train_dl=get_dataloaders(config=(monkeysee_config := update_config(
    #                 config=update_config_paths(
    #                     config=torch.load(os.path.join(monkeysee_ckpt_path, "generator.pt"), pickle_module=dill)["config"],
    #                     new_data_path=DATA_PATH,
    #                     replace_until_folder="csng",
    #                 ),
    #                 config_updates={
    #                     "data__cat_v1__dataset_config__batch_size": config["data"]["cat_v1"]["dataset_config"]["batch_size"],
    #                 }
    #             )
    #         ))[0]["train"]["cat_v1"],
    #         new_data_path=DATA_PATH,
    #     ),
    #     "use_data_config": monkeysee_config,
    #     "run_name": None,
    # },


    ### --- MindEye ---
    # "MindEye2 (B-6)": { # B-6
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_18-02-25_19-45", "subj06_reconstructions_zscored.pt"), pickle_module=dill),
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_b6__10-05-25_15-46", "subj06_test_reconstructions.pt"), pickle_module=dill), # seed=1
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_b6__11-05-25_16-21", "subj06_test_reconstructions.pt"), pickle_module=dill), # seed=2
    #         data_key="6",
    #         # zscore_reconstructions=True,
    #         zscore_reconstructions=False, # use for seed=0 and seed=1
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (B-6, post-hoc z-scored)": { # B-6
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_18-02-25_19-45", "subj06_reconstructions.pt"), pickle_module=dill),
    #         data_key="6",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (B-1-8)": { # B-1-8
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_19-02-25_16-52", "subj06_reconstructions_zscored.pt"), pickle_module=dill),
    #         data_key="6",
    #         zscore_reconstructions=False,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (B-1-8, post-hoc z-scored)": { # B-1-8
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_19-02-25_16-52", "subj06_reconstructions.pt"), pickle_module=dill)["MindEye2"]["stim_pred_best"][0],
    #         data_key="6",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (B-1-8 -> B-6)": { # B-6 fine-tuned from B-1-8
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_26-02-25_10-53", "subj06_reconstructions.pt"), pickle_module=dill)["MindEye2"]["stim_pred_best"][0],
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_b6__15-05-25_11-44", "subj06_test_reconstructions.pt"), pickle_module=dill), # seed=1
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_b6__13-05-25_12-45", "subj06_test_reconstructions_img2imgtimepoint5.pt"), pickle_module=dill), # seed=2
    #         data_key="6",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2": { # M-1
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_mouse_v1__24-03-25_22-50", "subj21067-10-18_reconstructions_zscored.pt"), pickle_module=dill),
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__10-05-25_15-54", "subj21067-10-18_test_reconstructions_zscored.pt"), pickle_module=dill), # seed=1
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__11-05-25_03-54", "subj21067-10-18_test_reconstructions_zscored.pt"), pickle_module=dill), # seed=2
    #         data_key="21067-10-18",
    #         zscore_reconstructions=False,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (post-hoc z-scored)": { # M-1
    #     "decoder": SavedReconstructionsDecoder(
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_mouse_v1__24-03-25_22-50", "subj21067-10-18_reconstructions.pt"), pickle_module=dill),
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__10-05-25_15-54", "subj21067-10-18_test_reconstructions.pt"), pickle_module=dill), # seed=1
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__11-05-25_03-54", "subj21067-10-18_test_reconstructions.pt"), pickle_module=dill), # seed=2
    #         data_key="21067-10-18",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (M-All)": { # M-All
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_mouse_v1__24-03-25_12-08", "subj21067-10-18_reconstructions_zscored.pt"), pickle_module=dill),
    #         data_key="21067-10-18",
    #         zscore_reconstructions=False,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (M-All, post-hoc z-scored)": { # M-All
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_mouse_v1__24-03-25_12-08", "subj21067-10-18_reconstructions.pt"), pickle_module=dill),
    #         data_key="21067-10-18",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (M-All -> M-1)": { # M-1 fine-tuned from M-All (csng_mouse_v1__24-03-25_12-08)
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_mouse_v1__03-04-25_16-06", "subj21067-10-18_test_reconstructions.pt"), pickle_module=dill),
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__13-05-25_12-13", "subj21067-10-18_test_reconstructions.pt"), pickle_module=dill), # seed=1
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_m1__13-05-25_12-30", "subj21067-10-18_test_reconstructions.pt"), pickle_module=dill), # seed=2
    #         data_key="21067-10-18",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2": { # cat v1
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_cat_v1__08-03-25_11-24", "subjcat_v1_reconstructions_zscored.pt"), pickle_module=dill),
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_cat_v1__10-05-25_15-36", "subjcat_v1_test_reconstructions_zscored.pt"), pickle_module=dill), # seed=1
    #         # reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_cat_v1__13-05-25_02-18", "subjcat_v1_test_reconstructions_zscored.pt"), pickle_module=dill), # seed=2
    #         data_key="cat_v1",
    #         zscore_reconstructions=False,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },
    # "MindEye2 (post-hoc z-scored)": { # cat v1
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_cat_v1__08-03-25_11-24", "subjcat_v1_reconstructions.pt"), pickle_module=dill),
    #         data_key="cat_v1",
    #         zscore_reconstructions=True,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },

    ### --- Final GAN-MEI ---
    ## brainreader mouse ---
    # "GAN": {
    #     "run_name": "2025-04-03_02-35-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-03_02-35-59", "decoder.pt"),
    # },
    # "GAN seed=1": {
    #     "run_name": "2025-05-10_15-22-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_15-22-59", "ckpt", "decoder_290.pt"),
    # },
    # "GAN seed=2": {
    #     "run_name": "2025-05-10_15-24-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_15-24-02", "decoder.pt"),
    # },
    # "GAN (B-1-8 -> B-6)": {
    #     "run_name": "2025-04-27_19-08-08",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-27_19-08-08", "decoder.pt"),
    # },
    # "GAN (B-1-8 -> B-6) seed=1": {
    #     "run_name": "2025-05-14_02-51-17",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-14_02-51-17", "decoder.pt"),
    # },
    # "GAN (B-1-8 -> B-6) seed=2": {
    #     "run_name": "2025-05-14_02-53-15",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-14_02-53-15", "decoder.pt"),
    # },
    # "GAN, MEIs ablation": {
    #     "run_name": "2025-04-09_20-33-37",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-09_20-33-37", "decoder.pt"),
    # },
    # "GAN, NEs ablation": {
    #     "run_name": "2025-04-13_01-46-17",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-46-17", "decoder.pt"),
    # },
    # "GAN, SSIML ablation": {
    #     "run_name": "2025-04-13_01-43-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-43-50", "decoder.pt"),
    # },
    # "GAN, MEIS & SSIML ablation": {
    #     "run_name": "2025-04-21_18-34-46",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_18-34-46", "decoder.pt"),
    # },
    # "GAN, MEIS & SSIML & NEs ablation": {
    #     "run_name": "2025-04-21_18-36-27",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_18-36-27", "decoder.pt"),
    # },
    # "GAN, SSIML & NEs ablation": {
    #     "run_name": "2025-04-21_18-39-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_18-39-16", "decoder.pt"),
    # },
    # "GAN, MEIs & NEs ablation": {
    #     "run_name": "2025-04-26_10-50-31",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-26_10-50-31", "decoder.pt"),
    # },

    # "GAN": { # highres
    #     "run_name": "2025-07-25_08-13-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-13-34", "ckpt", "decoder_255.pt"),
    # },
    # "GAN": { # highres
    #     "run_name": "2025-07-25_08-40-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-40-48", "decoder.pt"),
    # },
    # "GAN": { # highres
    #     "run_name": "2025-07-25_08-17-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-17-34", "ckpt", "decoder_480.pt"),
    # },

    ## High-res 144x256
    # "2025-07-25_08-06-35": { # highres
    #     "run_name": "2025-07-25_08-06-35",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-06-35", "ckpt", "decoder_655.pt"),
    # },
    # "2025-07-25_08-13-34": { # highres
    #     "run_name": "2025-07-25_08-13-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-13-34", "ckpt", "decoder_805.pt"),
    # },
    # "2025-07-25_08-34-34": { # highres
    #     "run_name": "2025-07-25_08-34-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-34-34", "ckpt", "decoder_695.pt"),
    # },

    ## High-res 72x128
    # "GAN best": { # highres
    #     "run_name": "2025-07-26_06-46-23",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-26_06-46-23", "decoder.pt"),
    # },
    # "GAN second best": { # highres
    #     "run_name": "2025-07-26_18-48-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-26_18-48-34", "decoder.pt"),
    # },
    # "2025-07-26_18-48-34": { # highres
    #     "run_name": "2025-07-26_18-48-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-26_18-48-34", "ckpt", "decoder_595.pt"),
    # },
    # "2025-07-26_06-46-23": { # highres
    #     "run_name": "2025-07-26_06-46-23",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-26_06-46-23", "ckpt", "decoder_595.pt"),
    # },
    # "2025-07-25_08-17-34": { # highres
    #     "run_name": "2025-07-25_08-17-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-17-34", "ckpt", "decoder_595.pt"),
    # },
    # "2025-07-25_08-20-34": { # highres
    #     "run_name": "2025-07-25_08-20-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-20-34", "ckpt", "decoder_595.pt"),
    # },
    # "2025-07-25_08-40-48": { # highres
    #     "run_name": "2025-07-25_08-40-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-25_08-40-48", "ckpt", "decoder_595.pt"),
    # },

    ## LRFs
    # "2025-07-29_06-29-57": {
    #     "run_name": "2025-07-29_06-29-57",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_06-29-57", "decoder.pt"),
    # },
    # "2025-07-29_07-16-39": {
    #     "run_name": "2025-07-29_07-16-39",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_07-16-39", "decoder.pt"),
    # },

    ## MEIs
    # "2025-07-29_03-25-21": {
    #     "run_name": "2025-07-29_03-25-21",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_03-25-21", "decoder.pt"),
    # },
    # "2025-07-29_03-27-41": {
    #     "run_name": "2025-07-29_03-27-41",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_03-27-41", "decoder.pt"),
    # },
    # "2025-07-29_05-41-08": {
    #     "run_name": "2025-07-29_05-41-08",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_05-41-08", "decoder.pt"),
    # },

    ## NLI
    # "2025-07-28_21-15-25": {
    #     "run_name": "2025-07-28_21-15-25",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-28_21-15-25", "decoder.pt"),
    # },
    # "2025-07-28_22-40-21": {
    #     "run_name": "2025-07-28_22-40-21",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-28_22-40-21", "decoder.pt"),
    # },
    # "2025-07-29_23-03-21": {
    #     "run_name": "2025-07-29_07-25-12",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_23-03-21", "ckpt", "decoder_270.pt"),
    # },

    ## MEI noise
    # "2025-07-29_22-47-37": { # std=0.1
    #     "run_name": "2025-07-29_22-47-37",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-47-37", "ckpt", "decoder_290.pt"),
    # },
    # "2025-07-29_22-48-19": { # std=0.2
    #     "run_name": "2025-07-29_22-48-19",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-48-19", "ckpt", "decoder_290.pt"),
    # },
    # "2025-07-29_22-50-46": { # std=0.3
    #     "run_name": "2025-07-29_22-50-46",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-50-46", "ckpt", "decoder_290.pt"),
    # },
    # "2025-07-29_22-51-18": { # std=0.5
    #     "run_name": "2025-07-29_22-51-18",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-51-18", "ckpt", "decoder_285.pt"),
    # },
    # "2025-07-29_22-51-54": { # std=1
    #     "run_name": "2025-07-29_22-51-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-51-54", "ckpt", "decoder_285.pt"),
    # },
    # "2025-07-29_22-52-45": { # std=3
    #     "run_name": "2025-07-29_22-52-45",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-52-45", "ckpt", "decoder_285.pt"),
    # },
    # "2025-07-29_22-53-21": { # std=10
    #     "run_name": "2025-07-29_22-53-21",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-07-29_22-53-21", "ckpt", "decoder_285.pt"),
    # },
    # # ---

    ## sensorium mouse v1 ---
    # "GAN": {
    #     "run_name": "2025-03-24_10-06-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-03-24_10-06-33", "decoder.pt"),
    # },
    # "GAN seed=1": {
    #     "run_name": "2025-05-10_16-02-01",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_16-02-01", "decoder.pt"),
    # },
    # "GAN seed=2": {
    #     "run_name": "2025-05-10_20-04-01",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_20-04-01", "decoder.pt"),
    # },
    # "GAN (M-All -> M-1)": {
    #     "run_name": "2025-04-29_01-37-25",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-29_01-37-25", "decoder.pt"),
    # },
    # "GAN (M-All -> M-1) seed=1": {
    #     "run_name": "2025-05-14_02-30-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-14_02-30-11", "decoder.pt"),
    # },
    # "GAN (M-All -> M-1) seed=2": {
    #     "run_name": "2025-05-14_02-47-02",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-14_02-47-02", "decoder.pt"),
    # },
    # "GAN, MEIs ablation": {
    #     "run_name": "2025-04-09_20-26-12",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-09_20-26-12", "decoder.pt"),
    # },
    # "GAN, SSIML ablation": {
    #     "run_name": "2025-04-13_01-51-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-51-16", "decoder.pt"),
    # },
    # "GAN, NEs ablation": {
    #     "run_name": "2025-04-13_01-52-37",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-52-37", "decoder.pt"),
    # },
    # "GAN, MEIS & SSIML ablation": {
    #     "run_name": "2025-04-21_21-39-27",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_21-39-27", "decoder.pt"),
    # },
    # "GAN, MEIS & SSIML & NEs ablation": {
    #     "run_name": "2025-04-22_00-32-18",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_00-32-18", "decoder.pt"),
    # },
    # "GAN, SSIML & NEs ablation": {
    #     "run_name": "2025-04-22_01-00-20",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_01-00-20", "decoder.pt"),
    # },
    # "GAN, MEIs & NEs ablation": {
    #     "run_name": "2025-04-26_10-47-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-26_10-47-48", "decoder.pt"),
    # }, # ---

    ## cat v1 ---
    "GAN": {
        "run_name": "2025-04-25_20-19-46",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_20-19-46", "ckpt", "decoder_195.pt"),
    },
    # "GAN seed=1": {
    #     "run_name": "2025-05-10_15-32-21",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_15-32-21", "ckpt", "decoder_185.pt"),
    # },
    # "GAN seed=2": {
    #     "run_name": "2025-05-10_15-34-01",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-05-10_15-34-01", "ckpt", "decoder_185.pt"),
    # },
    # "GAN, MEIs ablation": {
    #     "run_name": "2025-04-10_00-07-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-10_00-07-04", "ckpt", "decoder_160.pt"),
    # },
    # "GAN, SSIML ablation": {
    #     "run_name": "2025-04-13_01-57-10",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-57-10", "ckpt", "decoder_200.pt"),
    # },
    # "GAN, NEs ablation": {
    #     "run_name": "2025-04-13_01-58-11",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-13_01-58-11", "ckpt", "decoder_200.pt"),
    # },
    # "GAN, MEIs & NEs ablation": {
    #     "run_name": "2025-04-22_01-18-19",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_01-18-19", "ckpt", "decoder_200.pt"),
    # },
    # "GAN, MEIs & SSIML & NEs ablation": {
    #     "run_name": "2025-04-22_11-40-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_11-40-07", "ckpt", "decoder_200.pt"),
    # },
    # "GAN, MEIs & SSIML ablation": {
    #     "run_name": "2025-04-22_11-41-56",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_11-41-56", "ckpt", "decoder_200.pt"),
    # },
    # "GAN, SSIML & NEs ablation": {
    #     "run_name": "2025-04-22_11-43-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-22_11-43-16", "ckpt", "decoder_200.pt"),
    # }, # ---


    ### --- Number of neurons vs performance ---
    # Cat v1
    # "GAN (100)": {
    #     "run_name": "2025-04-24_00-20-32",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_00-20-32", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-25_02-00-10",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_02-00-10", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-24_00-24-40",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_00-24-40", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (2500)": {
    #     "run_name": "2025-04-24_00-26-38",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_00-26-38", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (5000)": {
    #     "run_name": "2025-04-24_01-19-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_01-19-34", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (8000)": {
    #     "run_name": "2025-04-24_00-22-38",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_00-22-38", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (0.1%)": {
    #     "run_name": "2025-04-20_17-39-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_17-39-36", "ckpt", "decoder_250.pt"),
    # },
    # "GAN (1%)": {
    #     "run_name": "2025-04-19_00-39-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-19_00-39-16", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (2.5%)": {
    #     "run_name": "2025-04-19_00-40-13",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-19_00-40-13", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (5%)": {
    #     "run_name": "2025-04-19_00-41-12",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-19_00-41-12", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (10%)": {
    #     "run_name": "2025-04-18_19-11-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_19-11-16", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2025-04-18_15-35-30",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_15-35-30", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2025-04-18_15-34-49",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_15-34-49", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2025-04-18_00-10-51",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_00-10-51", "ckpt", "decoder_200.pt"),
    # },
    # "GAN (100%)": {
    #     "run_name": "2025-03-22_18-54-55",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-03-22_18-54-55", "ckpt", "decoder_120.pt"),
    # },
    
    ## Brainreader mouse
    # "GAN (100)": {
    #     "run_name": "2025-04-24_20-39-26",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_20-39-26", "decoder.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-25_02-34-09",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_02-34-09", "decoder.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-24_12-19-37",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_12-19-37", "decoder.pt"),
    # },
    # "GAN (2500)": {
    #     "run_name": "2025-04-24_12-17-27",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_12-17-27", "decoder.pt"),
    # },
    # "GAN (5000)": {
    #     "run_name": "2025-04-24_12-18-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-24_12-18-36", "decoder.pt"),
    # },
    # "GAN (8000)": {
    #     "run_name": "2025-04-25_01-21-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_01-21-04", "decoder.pt"),
    # },
    # "GAN (1%)": {
    #     "run_name": "2025-04-18_15-30-07",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_15-30-07", "decoder.pt"),
    # },
    # "GAN (1.5%)": {
    #     "run_name": "2025-04-20_17-36-03",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_17-36-03", "decoder.pt"),
    # },
    # "GAN (2.5%)": {
    #     "run_name": "2025-04-18_15-29-26",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_15-29-26", "decoder.pt"),
    # },
    # "GAN (5%)": {
    #     "run_name": "2025-04-21_17-34-13",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_17-34-13", "decoder.pt"),
    # },
    # "GAN (10%)": {
    #     "run_name": "2025-04-20_16-59-30",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_16-59-30", "decoder.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2025-04-20_16-58-04",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_16-58-04", "decoder.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2025-04-18_10-35-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_10-35-48", "decoder.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2025-04-18_10-35-03",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-18_10-35-03", "decoder.pt"),
    # },
    # "GAN (100%)": {
    #     "run_name": "2025-04-03_02-35-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-03_02-35-59", "decoder.pt"),
    # },

    ## Sensorium mouse v1
    # "GAN (100)": {
    #     "run_name": "2025-04-25_13-24-22",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_13-24-22", "decoder.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-25_15-29-31",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_15-29-31", "decoder.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-25_13-17-18",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_13-17-18", "decoder.pt"),
    # },
    # "GAN (2500)": {
    #     "run_name": "2025-04-25_11-36-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_11-36-48", "decoder.pt"),
    # },
    # "GAN (5000)": {
    #     "run_name": "2025-04-25_10-40-46",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_10-40-46", "decoder.pt"),
    # },
    # "GAN (8000)": {
    #     "run_name": "2025-04-25_13-43-13",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-25_13-43-13", "decoder.pt"),
    # },
    # "GAN (1%)": {
    #     "run_name": "2025-04-20_02-21-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_02-21-14", "decoder.pt"),
    # },
    # "GAN (1.5%)": {
    #     "run_name": "2025-04-20_17-37-12",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_17-37-12", "decoder.pt"),
    # },
    # "GAN (2.5%)": {
    #     "run_name": "2025-04-20_11-33-45",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_11-33-45", "decoder.pt"),
    # },
    # "GAN (5%)": {
    #     "run_name": "2025-04-20_02-23-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_02-23-33", "decoder.pt"),
    # },
    # "GAN (10%)": {
    #     "run_name": "2025-04-20_02-20-19",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_02-20-19", "decoder.pt"),
    # },
    # "GAN (25%)": {
    #     "run_name": "2025-04-20_02-19-43",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_02-19-43", "decoder.pt"),
    # },
    # "GAN (50%)": {
    #     "run_name": "2025-04-20_02-18-41",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-20_02-18-41", "decoder.pt"),
    # },
    # "GAN (75%)": {
    #     "run_name": "2025-04-19_23-25-13",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-19_23-25-13", "decoder.pt"),
    # },
    # "GAN (100%)": {
    #     "run_name": "2025-03-24_10-06-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-03-24_10-06-33", "decoder.pt"),
    # },


    ### --- Amount of data vs performance ---
    # # Cat v1
    # "GAN (100)": {
    #     "run_name": "2025-04-21_01-27-39",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-27-39", "decoder.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-21_01-29-43",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-29-43", "decoder.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-21_01-26-23",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-26-23", "decoder.pt"),
    # },
    # "GAN (2000)": {
    #     "run_name": "2025-04-21_13-59-50",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_13-59-50", "decoder.pt"),
    # },
    # "GAN (3000)": {
    #     "run_name": "2025-04-21_01-28-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-28-59", "decoder.pt"),
    # },
    # "GAN (4000)": {
    #     "run_name": "2025-04-21_14-02-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_14-02-14", "decoder.pt"),
    # },
    # "GAN (5000)": {
    #     "run_name": "2025-04-21_14-04-43",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_14-04-43", "decoder.pt"),
    # },
    # "GAN (45000)": {
    #     "run_name": "2025-03-22_18-54-55",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-03-22_18-54-55", "ckpt", "decoder_120.pt"),
    # },

    # Brainreader mouse
    # "GAN (100)": {
    #     "run_name": "2025-04-21_01-58-49",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-58-49", "decoder.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-21_11-00-26",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-00-26", "decoder.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-21_01-31-35",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_01-31-35", "decoder.pt"),
    # },
    # "GAN (2000)": {
    #     "run_name": "2025-04-21_11-02-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-02-33", "decoder.pt"),
    # },
    # "GAN (3000)": {
    #     "run_name": "2025-04-21_11-01-28",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-01-28", "decoder.pt"),
    # },
    # "GAN (4500)": {
    #     "run_name": "2025-04-03_02-35-59",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-03_02-35-59", "decoder.pt"),
    # },

    # Sensorium mouse v1
    # "GAN (100)": {
    #     "run_name": "2025-04-21_13-27-16",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_13-27-16", "decoder.pt"),
    # },
    # "GAN (500)": {
    #     "run_name": "2025-04-21_12-20-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_12-20-54", "decoder.pt"),
    # },
    # "GAN (1000)": {
    #     "run_name": "2025-04-21_11-06-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-06-14", "decoder.pt"),
    # },
    # "GAN (2000)": {
    #     "run_name": "2025-04-21_11-05-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-05-36", "decoder.pt"),
    # },
    # "GAN (3000)": {
    #     "run_name": "2025-04-21_11-04-34",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-21_11-04-34", "decoder.pt"),
    # },
    # "GAN (4473)": {
    #     "run_name": "2025-03-24_10-06-33",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-03-24_10-06-33", "decoder.pt"),
    # },



    ### --- Inverted Encoder Decoder ---
    # "Inverted Encoder-Decoder": { # sensorium mouse v1
    #     "decoder": InvertedEncoderDecoder(
    #         encoder=get_encoder_sensorium_mouse_v1(
    #             ckpt_path=os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    #             eval_mode=True,
    #             device=config["device"],
    #         ),
    #         decoder=load_decoder_from_ckpt(
    #             ckpt_path=os.path.join(DATA_PATH, "models", "gan", "2025-03-22_15-57-43", "decoder.pt"),
    #             load_best=True,
    #             load_only_core=False,
    #             strict=True,
    #             device=config["device"],
    #         )[0],
    #         img_dims=(1, 36, 64),
    #         stim_pred_init="decoder",
    #         opter_config={"lr": 50},
    #         n_steps=1000,
    #         img_grad_gauss_blur_config={"kernel_size": 13, "sigma": 1.5},
    #         device=config["device"],
    #     ),
    #     "run_name": None,
    # },

    ## multiple ---
    # "GAN (B-1-8 + M-All + C)": {
    #     "run_name": "2025-02-26_00-24-36",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-26_00-24-36", "ckpt", "decoder_120.pt"),
    # },
    # "GAN (B-All + M-All + C)": {
    #     "run_name": "2025-02-26_00-20-14",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-26_00-20-14", "ckpt", "decoder_60.pt"),
    # },

    ### --- CNN MSE ---
    # "CNN": {
    #     "run_name": "2024-12-17_03-20-48",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "cnn", "2024-12-17_03-20-48", "decoder.pt"),
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


    ### --- Ablation studies ---
    # "GAN": {
    #     "run_name": "2025-02-15_23-28-30",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-15_23-28-30", "decoder.pt"),
    # },
    # "GAN, all-ones MEIs": {
    #     "run_name": "2025-02-15_13-31-01",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-15_13-31-01", "decoder.pt"),
    # },
    # "GAN, trainable MEIs": {
    #     "run_name": "2025-02-16_16-36-57",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-16_16-36-57", "decoder.pt"),
    # },
    # "GAN, BDfR L2": {
    #     "run_name": "2025-02-15_16-43-09",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-15_16-43-09", "decoder.pt"),
    # },
    # "GAN, BDfR L1": {
    #     "run_name": "2025-02-15_16-45-54",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-15_16-45-54", "decoder.pt"),
    # },
    # "GAN, BDfS L1": {
    #     "run_name": "2025-02-15_16-48-29",
    #     "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-02-15_16-48-29", "decoder.pt"),
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

    ### get sample data
    s = get_sample_data(dls=get_dataloaders(config=cfg)[0], config=cfg, sample_from_tier="test")
    stim, resp, sample_dataset, sample_data_key = s["stim"].to(cfg["device"]), s["resp"].to(cfg["device"]), s["sample_dataset"], s["sample_data_key"]

    ### load previous comparison results
    runs_to_compare = dict()
    if cfg["comparison"]["load_ckpt"] is not None:
        print(f"[INFO] Loading checkpoint from {cfg['comparison']['load_ckpt']['path']}...")
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

    ### load metrics
    inp_zscored = check_if_data_zscored(cfg=cfg)
    _get_metrics_load_brain_distance_with_cfg = None
    if "BrainDistance" in cfg["comparison"]["losses_to_plot"]:
        assert len(cfg["crop_wins"].keys()) == 1, "BrainDistance only implemented for testing on single-subject data."
        if "brainreader_mouse" in cfg["data"]:
            assert cfg["data"]["brainreader_mouse"]["sessions"] == [6], "BrainDistance only implemented for testing on single-subject data of 6."
            _encoder = get_encoder_brainreader(
                os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
                device=config["device"],
            )
            pad_stim_pred_to = None
        elif "mouse_v1" in cfg["data"]:
            assert len(cfg["data"]["mouse_v1"]["dataset_config"]["paths"]) == 1, "BrainDistance only implemented for testing on single-subject data."
            _encoder = get_encoder_sensorium_mouse_v1(
                os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
                device=config["device"],
            )
            pad_stim_pred_to = (1, 1, 36, 64)
        elif "cat_v1" in cfg["data"]:
            _encoder = get_encoder_cat_v1(
                os.path.join(DATA_PATH, "models", "encoders", "encoder_c.pt"),
                device=config["device"],
            )
            pad_stim_pred_to = (1, 1, 50, 50)
        else:
            raise ValueError("BrainDistance only implemented for testing on single-subject data.")
        _get_metrics_load_brain_distance_with_cfg={
            "encoder": _encoder,
            "use_gt_resp": True,
            "resp_loss_fn": F.mse_loss,
            "zscore_inp": inp_zscored is False,
            "minmax_normalize_inp": False,
            "pad_stim_pred_to": pad_stim_pred_to,
            "device": cfg["device"],
        }
    metrics = {
        data_key: get_metrics(
            inp_zscored=inp_zscored,
            crop_win=cfg["crop_wins"][data_key],
            load_brain_distance_with_cfg=_get_metrics_load_brain_distance_with_cfg,
            device=cfg["device"],
        ) for data_key in cfg["crop_wins"].keys()
    }

    ### load and compare models
    for k in runs_to_compare.keys():
        print(f"\n-----\n[INFO] Loading {k} model from ckpt (run name: {runs_to_compare[k]['run_name']})...")
        ### check if already loaded
        if "test_losses" in runs_to_compare[k] \
            and np.all([loss_name in runs_to_compare[k]["test_losses"][0]["total"] for loss_name in cfg["comparison"]["losses_to_plot"]]):
            print(f"[INFO] Skipping (evaluation results already present)...")
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
                print(f"[INFO] Finding the best ckpt out of {len(run_dict['ckpt_paths'])} according to {cfg['comparison']['find_best_ckpt_according_to']}...")
                get_val_dl_fn = lambda: get_dataloaders(config=cfg)[0]["val"]
                run_dict["ckpt_paths"] = [find_best_ckpt(get_dl_fn=get_val_dl_fn, config=cfg, ckpt_paths=run_dict["ckpt_paths"], metrics=metrics)[0]]
                print(f"[INFO] Best checkpoint found: {run_dict['ckpt_paths'][0]}")

        ### eval ckpts
        print(f"[INFO] Evaluating checkpoints on the test set...")
        for ckpt_path in run_dict["ckpt_paths"]:
            ### get decoder
            if "decoder" in run_dict and run_dict["decoder"] is not None:
                print(f"[INFO] Using {k} model from run_dict...")
                decoder = run_dict["decoder"]
                ckpt = None
            else:
                ### load ckpt and init
                decoder, ckpt = load_decoder_from_ckpt(ckpt_path=ckpt_path, device=cfg["device"], load_best=cfg["comparison"]["load_best"], load_only_core=False, strict=True)
                run_dict["configs"].append(ckpt["config"])
                run_dict["histories"].append(ckpt["history"])
                run_dict["best_val_losses"].append(ckpt["best"]["val_loss"])
            decoder.eval()

            ### get data samples for plotting and eval
            seed_all(cfg["seed"])
            if run_dict.get("use_data_config", None) is not None:
                ### prevent mismatching data
                assert "brainreader" not in run_dict["use_data_config"]["data"] or "brainreader" in cfg["data"], \
                    "Brainreader data must be present in the main config."
                assert "mouse_v1" not in run_dict["use_data_config"]["data"] or "mouse_v1" in cfg["data"], \
                    "Mouse V1 data must be present in the main config."
                assert "cat_v1" not in run_dict["use_data_config"]["data"] or "cat_v1" in cfg["data"], \
                    "Cat V1 data must be present in the main config."
                assert "brainreader" not in run_dict["use_data_config"]["data"] or run_dict["use_data_config"]["data"]["brainreader"]["sessions"] == cfg["data"]["brainreader"]["sessions"], \
                    "Brainreader sessions must be the same for the comparison across all runs."
                assert "mouse_v1" not in run_dict["use_data_config"]["data"] or run_dict["use_data_config"]["data"]["mouse_v1"]["dataset_config"]["paths"] == cfg["data"]["mouse_v1"]["dataset_config"]["paths"], \
                    "Mouse V1 sessions (dataset_config.paths) must be the same for the comparison across all runs."

                ### data samples
                dls, neuron_coords = get_dataloaders(config=run_dict["use_data_config"])
                s = get_sample_data(dls=dls, config=run_dict["use_data_config"], sample_from_tier="test")
                stim, resp, sample_dataset, sample_data_key = s["stim"].to(cfg["device"]), s["resp"].to(cfg["device"]), s["sample_dataset"], s["sample_data_key"]

                ### eval data
                cfg_for_eval_dls = run_dict["use_data_config"]
            else:
                ### data samples
                dls, neuron_coords = get_dataloaders(config=cfg)
                s = get_sample_data(dls=dls, config=cfg, sample_from_tier="test")
                stim, resp, sample_dataset, sample_data_key = s["stim"].to(cfg["device"]), s["resp"].to(cfg["device"]), s["sample_dataset"], s["sample_data_key"]

                cfg_for_eval_dls = cfg

            ### get sample reconstructions
            stim_pred_best = dict()
            if "brainreader_mouse" in cfg["data"]:
                stim_pred_best[s["b_sample_data_key"]] = decoder(s["b_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["b_sample_dataset"]][s["b_sample_data_key"]], data_key=s["b_sample_data_key"]).detach().cpu()
            if "cat_v1" in cfg["data"]:
                stim_pred_best[s["c_sample_data_key"]] = decoder(s["c_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["c_sample_dataset"]], data_key=s["c_sample_data_key"]).detach().cpu()
            if "mouse_v1" in cfg["data"]:
                stim_pred_best[s["m_sample_data_key"]] = decoder(s["m_resp"].to(cfg["device"]), neuron_coords=neuron_coords[s["m_sample_dataset"]][s["m_sample_data_key"]], pupil_center=s["m_pupil_center"].to(cfg["device"]), data_key=s["m_sample_data_key"]).detach().cpu()
            if cfg["comparison"]["max_n_reconstruction_samples"] is not None:
                for k in stim_pred_best.keys():
                    stim_pred_best[k] = stim_pred_best[k][:cfg["comparison"]["max_n_reconstruction_samples"]]
            run_dict["stim_pred_best"].append(stim_pred_best)
            if isinstance(decoder, SavedReconstructionsDecoder):
                decoder.reset_counter()

            ### eval
            eval_dls, _ = get_dataloaders(config=cfg_for_eval_dls)
            seed_all(cfg["seed"])
            run_dict["test_losses"].append(eval_decoder(
                model=decoder,
                dataloaders=eval_dls[cfg["comparison"]["eval_tier"]],
                loss_fns=metrics,
                crop_wins=cfg["crop_wins"],
                eval_every_n_samples=cfg["comparison"]["eval_every_n_samples"],
                z_score_wrt_target=cfg["comparison"]["z_score_wrt_target"],
            ))

            ### collect all preds and targets
            if cfg["comparison"]["save_all_preds_and_targets"]:
                eval_dls, _ = get_dataloaders(config=cfg_for_eval_dls)
                seed_all(cfg["seed"])
                run_dict["all_preds"], run_dict["all_targets"] = collect_all_preds_and_targets(
                    model=decoder,
                    dataloaders=eval_dls[cfg["comparison"]["eval_tier"]],
                    crop_wins=cfg["crop_wins"],
                    device=cfg["device"],
                )
        print("-----\n")

    ### save the results
    if cfg["comparison"]["save_dir"]:
        print(f"[INFO] Saving the results to {cfg['comparison']['save_dir']}")
        os.makedirs(cfg["comparison"]["save_dir"], exist_ok=True)
        torch.save({
                "runs": runs_to_compare,
                "config": cfg,
            }, os.path.join(cfg["comparison"]["save_dir"], f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"),
            pickle_module=dill,
        )

    ### plot reconstructions
    print(f"[INFO] Plotting reconstructions...")
    for f_type in ("png", "pdf"):
        for data_key in cfg["crop_wins"].keys():
            plot_reconstructions(
                runs=runs_to_compare,
                stim=stim[:cfg["comparison"]["max_n_reconstruction_samples"]] if cfg["comparison"]["max_n_reconstruction_samples"] is not None else stim,
                stim_label="Target",
                data_key=data_key,
                crop_win=cfg["crop_wins"][data_key],
                save_to=os.path.join(
                    cfg["comparison"]["save_dir"],
                    f"reconstructions_{data_key}.{f_type}"
                ) if cfg["comparison"]["save_dir"] else None,
            )

    ### plot metrics
    print(f"[INFO] Plotting metrics...")
    for f_type in ("png", "pdf"):
        plot_metrics(
            runs_to_compare=runs_to_compare,
            losses_to_plot=cfg["comparison"]["losses_to_plot"],
            save_to=os.path.join(
                cfg["comparison"]["save_dir"],
                f"metrics.{f_type}"
            ) if cfg["comparison"]["save_dir"] else None,
        )


if __name__ == "__main__":
    run_comparison(cfg=config)

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
from csng.models.utils.energy_guided_diffusion import EGGDecoder
from csng.utils.mix import seed_all, check_if_data_zscored, update_config_paths, update_config
from csng.utils.data import standardize, normalize, crop
from csng.utils.comparison import find_best_ckpt, load_decoder_from_ckpt, plot_reconstructions, plot_metrics, eval_decoder, SavedReconstructionsDecoder, collect_all_preds_and_targets
from csng.losses import get_metrics
from csng.data import get_dataloaders, get_sample_data
from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.mouse_v1.encoder import get_encoder as get_encoder_sensorium_mouse_v1
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1

from monkeysee.SpatialBased.decoding_wrapper import MonkeySeeDecoder
from cae.model import CAEDecoder

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
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 36,
    "sessions": [6],
    "drop_last": True,
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
#         "batch_size": 12,
#         # "batch_size": 36,
#         "stim_keys": ("stim",),
#         "resp_keys": ("exc_resp", "inh_resp"),
#         "return_coords": True,
#         "return_ori": False,
#         "coords_ori_filepath": os.path.join(DATA_PATH_CAT_V1, "pos_and_ori.pkl"),
#         "cached": False,
#         "stim_normalize_mean": 46.143,
#         "stim_normalize_std": 24.960,
#         "resp_normalize_mean": None,
#         "resp_normalize_std": torch.load(
#             os.path.join(DATA_PATH_CAT_V1, "responses_std.pt")
#         ),
#     },
# }
# add crop_wins for cat v1 data
# config["crop_wins"]["cat_v1"] = config["data"]["cat_v1"]["crop_win"]

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
    "save_all_preds_and_targets": True,
    "save_dir": None,
    "save_dir": os.path.join(
        # DATA_PATH,
        "results",
        "b6",
    ),
    "load_ckpt": None,
    "losses_to_plot": [
        "SSIM",
        "PixCorr",
        "Alex(2)",
        "Alex(5)",
        "Incep",
        "CLIP",
        "Eff",
        "SwAV",
    ],
}

### methods to compare
config["comparison"]["to_compare"] = {
    ### --- MEIcoder ---
    "MEIcoder": {
        "run_name": "2025-04-03_02-35-59",
        "ckpt_path": os.path.join(DATA_PATH, "models", "gan", "2025-04-03_02-35-59", "decoder.pt"),
    },

    ### --- CAEDecoder ---
    # "CAE": {
    #     "decoder": CAEDecoder(
    #         ckpt_path=os.path.join(DATA_PATH_CAE, "runs", "27-07-2025_15-13", "best_model.pt"),
    #     ).to(config["device"]),
    #     "run_name": None,
    # },

    ### --- Inverted encoder ---
    # "Inverted Encoder": {
    #     "decoder": EnsembleInvEnc(
    #         encoder_paths=[
    #             os.path.join(DATA_PATH, "models", "encoders", "encoder_b6.pt"),
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

    ### --- Energy guided diffusion ---
    # "EGG": {
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

    ### --- MonkeySee ---
    # "MonkeySee": {
    #     "decoder": MonkeySeeDecoder(
    #         ckpt_dir=(monkeysee_ckpt_path := os.path.join(DATA_PATH, "monkeysee", "runs", "18-02-2025_19-32")),
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

    ### --- MindEye ---
    # "MindEye2 (B-6)": {
    #     "decoder": SavedReconstructionsDecoder(
    #         reconstructions=torch.load(os.path.join(DATA_PATH, "mindeye", "evals", "csng_18-02-25_19-45", "subj06_reconstructions_zscored.pt"), pickle_module=dill),
    #         data_key="6",
    #         # zscore_reconstructions=True,
    #         zscore_reconstructions=False,
    #         device=config["device"],
    #     ),
    #     "run_name": None,
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

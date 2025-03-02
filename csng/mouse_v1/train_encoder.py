import os
import numpy as np
import json
import dill
import torch
from collections import OrderedDict
from nnfabrik.builder import get_model, get_trainer
from sensorium.utility import get_correlations
from sensorium.utility.scores import get_poisson_loss
from sensorium.utility.measure_helpers import get_df_for_scores

from csng.utils.mix import seed_all
from csng.mouse_v1.data import get_mouse_v1_dataloaders


DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")


### setup config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "save_path": os.path.join(DATA_PATH, "models", "encoders", "encoder_m1.pt"),
    # "load_ckpt": os.path.join(DATA_PATH, "models", "encoders", "encoder_mall.pt"),
    "train": True,
}

### mouse v1 data
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
        "batch_size": 64,
        "drop_last": True,
        "seed": config["seed"],
        "use_cache": False,
    },
    "skip_train": False,
    "skip_val": False,
    "skip_test": False,
    "normalize_neuron_coords": True,
    "average_test_multitrial": True,
    "save_test_multitrial": True,
    "test_batch_size": 64,
    "device": config["device"],
}

### model
_dls = get_mouse_v1_dataloaders(config=config)[0]
config["model_fn"] = "sensorium.models.stacked_core_full_gauss_readout"
config["model_config"] = {
    ### cobos et al. 2022?
    # "layers": 3,
    # "input_kern": 15,
    # "hidden_channels": 32,
    # "depth_separable": False,
    # "laplace_pyramid": True,

    "pad_input": False,
    "layers": 4,
    "input_kern": 9,
    "gamma_input": 6.3831,
    "gamma_readout": 0.0076,
    "hidden_kern": 7,
    "hidden_channels": 64,
    "hidden_padding": 3,
    "depth_separable": True,
    # "grid_mean_predictor": None, # neuron coords are not available
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 1,
        "hidden_features": 30,
        "final_tanh": True,
    },
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
    "shifter": False,
    "stack": -1,
    "mean_activity_dict": {
        data_key: torch.from_numpy(dat.statistics["responses"]["all"]["mean"]).to(config["device"])
        for data_key, dat in zip(_dls["mouse_v1"]["train"].data_keys, _dls["mouse_v1"]["train"].datasets)
    },
}
del _dls

### trainer config
config["trainer_fn"] = "sensorium.training.standard_trainer"
config["trainer_config"] = {
    "max_iter": 100,
    "verbose": True,
    "lr_decay_steps": 4,
    "avg_loss": False,
    "lr_init": 0.009,
    "track_training": True,
    "weight_decay": 0.,
    "ckpt_path": os.path.join(DATA_PATH, "models", "encoder_m1_latest_ckpt.pt"),
}


### encoder training pipeline
def run_training(cfg):
    print(f"... Running on {cfg['device']} ...")
    seed_all(cfg["seed"])

    ### prepare dataloaders compatible w/ nnfabrik
    _dls = get_mouse_v1_dataloaders(config=cfg)[0]
    data_keys = _dls["mouse_v1"]["train"].data_keys
    dls = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dls["mouse_v1"]["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dls["mouse_v1"]["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dls["mouse_v1"]["test"].dataloaders)}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=cfg["model_fn"],
        model_config=cfg["model_config"],
        dataloaders=dls,
        seed=cfg["seed"],
    )

    ### load ckpt
    if cfg.get("load_ckpt", None) is not None:
        print(f"[INFO] Loading ckpt from {cfg['load_ckpt']}")
        model.load_state_dict(torch.load(cfg["load_ckpt"], pickle_module=dill)["model"])
    model.to(cfg["device"])
    print(f"[INFO] Config: {json.dumps(cfg, indent=2, default=str)}")

    ### train
    if cfg["train"]:
        seed_all(cfg["seed"])
        trainer = get_trainer(trainer_fn=cfg["trainer_fn"], trainer_config=cfg["trainer_config"])
        print(f"[INFO] Training starts...")
        seed_all(cfg["seed"])
        validation_score, trainer_output, state_dict = trainer(model, dls, seed=cfg["seed"])
        print(f"{trainer_output=}")
        print(f"{validation_score=}")

        ### save
        print(f"[INFO] Saving the model...")
        torch.save({
            "config": cfg,
            "model": model.state_dict(),
            "val_score": validation_score,
            "trainer_output": trainer_output,
            "state_dict": state_dict,
        }, cfg["save_path"], pickle_module=dill)

    ### evaluate
    model.eval()
    print(f"[INFO] Evaluating correlation to average...")
    seed_all(cfg["seed"])
    correlation_to_avg = get_correlations(model, dls, tier="test", device=cfg["device"], as_dict=True)
    df_corr_avg = get_df_for_scores(session_dict=correlation_to_avg, measure_attribute="Correlation to Average").groupby("dataset").mean()
    print(df_corr_avg)

    print(f"[INFO] Evaluating validation and test loss...")
    print("Validation loss: ", get_poisson_loss(
        model,
        dls["validation"],
        device=cfg["device"],
        as_dict=False,
        avg=True,
        per_neuron=False,
        eps=1e-12,
    ))
    print("Test loss: ", get_poisson_loss(
        model,
        dls["test"],
        device=cfg["device"],
        as_dict=False,
        avg=True,
        per_neuron=False,
        eps=1e-12,
    ))


if __name__ == "__main__":
    run_training(cfg=config)

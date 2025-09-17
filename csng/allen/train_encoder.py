import os
import numpy as np
import json
import dill
from pathlib import Path
import torch
from collections import OrderedDict
from nnfabrik.builder import get_model, get_trainer
from sensorium.utility import get_correlations
from sensorium.utility.scores import get_poisson_loss
from sensorium.utility.measure_helpers import get_df_for_scores

from csng.utils.mix import seed_all
from csng.allen.data import get_allen_dataloaders

DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAE = os.path.join(os.environ["DATA_PATH"], "cae")


### setup config
config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "save_path": os.path.join(DATA_PATH, "models", "encoder_allen.pt"),
    "train": True,
}

### allen
config["data"]["allen"] = {
    "device": config["device"],
    "val_split_seed": config["seed"],
    "mixing_strategy": "sequential",
    "batch_size": 32,
    # "batch_size": 4,
    "val_split_frac": 0.2,
    "zscore_images": True,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
}

### model
_dls = get_allen_dataloaders(config=config["data"]["allen"])
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
    "grid_mean_predictor": None, # neuron coords are not available

    # "pad_input": False,
    # "layers": 8,
    # "input_kern": 9,
    # "gamma_input": 6.3831,
    # "gamma_readout": 0.0076,
    # "hidden_kern": 7,
    # "hidden_channels": 128,
    # "hidden_padding": 3,
    # "depth_separable": True,
    # "grid_mean_predictor": None, # neuron coords are not available

    # "grid_mean_predictor": {
    #     "type": "cortex",
    #     "input_dimensions": 2,
    #     "hidden_layers": 1,
    #     "hidden_features": 30,
    #     "final_tanh": True,
    # },
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
    "shifter": False,
    "stack": -1,
    # "mean_activity_dict": {
    #     data_key: torch.from_numpy(np.load(
    #         os.path.join(Path(dset.dataset_dir).parent.absolute(), "stats", "responses_mean.npy"))
    #     ).to(config["device"])
    #     for data_key, dset in zip(_dls["brainreader_mouse"]["train"].data_keys, _dls["brainreader_mouse"]["train"].datasets)
    # },
}
del _dls

### trainer config
config["trainer_fn"] = "sensorium.training.standard_trainer"
config["trainer_config"] = {
    "max_iter": 1,
    "verbose": True,
    "lr_decay_steps": 4,
    "avg_loss": False,
    "lr_init": 0.009,
    "track_training": True,
    "weight_decay": 0.,
    "ckpt_path": os.path.join(DATA_PATH, "models", "encoder_latest_ckpt.pt"),
}



### encoder training pipeline
def run_training(cfg):
    print(f"... Running on {cfg['device']} ...")
    seed_all(cfg["seed"])

    ### prepare dataloaders compatible w/ nnfabrik
    _dls = get_allen_dataloaders(config=cfg["data"]["allen"])["allen"]
    dls = OrderedDict({
        "train": OrderedDict({"allen": _dls["train"]}),
        "validation": OrderedDict({"allen": _dls["val"]}),
        "test": OrderedDict({"allen": _dls["test"]}),
    })

    ### build the encoder model
    seed_all(cfg["seed"])
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

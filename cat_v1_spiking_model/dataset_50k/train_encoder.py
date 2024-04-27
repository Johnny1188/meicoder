import os
import random
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
import dill
import torch
from collections import OrderedDict
import lovely_tensors as lt
from nnfabrik.builder import get_model, get_trainer
from sensorium.utility import get_correlations
from sensorium.utility.measure_helpers import get_df_for_scores

from cat_v1_spiking_model.dataset_50k.data import prepare_v1_dataloaders

lt.monkey_patch()

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")
print(f"{DATA_PATH=}")


config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
    },
    # "crop_win": (slice(15, 35), slice(15, 35)),
    "crop_win": (20, 20),
    "only_cat_v1_eval": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    # "load_ckpt": os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter.pth"),
}
config["data"]["cat_v1"] = {
    "train_path": os.path.join(DATA_PATH, "datasets", "train"),
    "val_path": os.path.join(DATA_PATH, "datasets", "val"),
    "test_path": os.path.join(DATA_PATH, "datasets", "test"),
    "image_size": [50, 50],
    "crop": False,
    "batch_size": 128,
    "stim_keys": ("stim",),
    "resp_keys": ("exc_resp", "inh_resp"),
    "return_coords": True,
    "return_ori": False,
    "coords_ori_filepath": os.path.join(DATA_PATH, "pos_and_ori.pkl"),
    "cached": False,
    "stim_normalize_mean": 46.143,
    "stim_normalize_std": 20.420,
    "resp_normalize_mean": torch.load(
        os.path.join(DATA_PATH, "responses_mean.pt")
    ),
    "resp_normalize_std": torch.load(
        os.path.join(DATA_PATH, "responses_std.pt")
    ),
    "clamp_neg_resp": True,
}

class Neurons:
    def __init__(self, coords):
        self.cell_motor_coordinates = coords


if __name__ == "__main__":
    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = dict()
    _dataloaders["cat_v1"] = prepare_v1_dataloaders(**config["data"]["cat_v1"])
    for k in _dataloaders["cat_v1"].keys():
        _dataloaders["cat_v1"][k].dataset.neurons = Neurons(
            coords=_dataloaders["cat_v1"][k].dataset.coords["all"].numpy()
        )
    dataloaders = OrderedDict({
        "train": OrderedDict({"cat_v1": _dataloaders["cat_v1"]["train"]}),
        "validation": OrderedDict({"cat_v1": _dataloaders["cat_v1"]["val"]}),
        "test": OrderedDict({"cat_v1": _dataloaders["cat_v1"]["test"]}),
    })

    ### build the encoder model
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {
        'pad_input': False,
        'layers': 4,
        'input_kern': 9,
        'gamma_input': 6.3831,
        'gamma_readout': 0.0076,
        'hidden_kern': 7,
        'hidden_channels': 64,
        'depth_separable': True,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True,
        },
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'shifter': False,
        'stack': -1,
    }
    model = get_model(
        model_fn=model_fn,
        model_config=model_config,
        dataloaders=dataloaders,
        seed=config["seed"],
    )

    ### load ckpt
    if config["load_ckpt"] is not None:
        print(f"[INFO] Loading ckpt from {config['load_ckpt']}")
        model.load_state_dict(torch.load(config["load_ckpt"], pickle_module=dill)["model"])

    ### train
    model.to(config["device"])
    trainer_fn = "sensorium.training.standard_trainer"
    trainer_config = {
        'max_iter': 200,
        'verbose': True,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.008,
        "track_training": True,
    }
    trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=trainer_config)
    print(
        f"[INFO] Config:"
        f"\n base config={json.dumps(config, indent=2, default=str)}"
        f"\n\n model config={json.dumps(model_config, indent=2, default=str)}"
        f"\n\n trainer config={json.dumps(trainer_config, indent=2, default=str)}\n"
    )
    print(f"[INFO] Training starts...")
    validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=config["seed"])
    print(f"{trainer_output=}")
    print(f"{validation_score=}")
    
    print(f"[INFO] Evaluating single trial correlation...")
    model.eval()
    single_trial_correlation = get_correlations(model, dataloaders, tier="test", device=config["device"], as_dict=True)
    df_corr_mean = get_df_for_scores(session_dict=single_trial_correlation, measure_attribute="Single Trial Correlation").groupby("dataset").mean()
    print(df_corr_mean)

    ### save
    torch.save({
            "config": config,
            "model_fn": model_fn,
            "model_config": model_config,
            "trainer_fn": trainer_fn,
            "trainer_config": trainer_config,
            "model": model.state_dict(),
            "val_score": validation_score,
            "trainer_output": trainer_output,
            "test_single_trial_corr": df_corr_mean,
            "state_dict": state_dict,
        }, "encoder_cat_v1_no_shifter.pth", pickle_module=dill,
    )

import os
import numpy as np
import json
import dill
import torch
from collections import OrderedDict
from nnfabrik.builder import get_data, get_model, get_trainer
from sensorium.utility import get_correlations, get_signal_correlations
from sensorium.utility.scores import get_poisson_loss
from sensorium.utility.measure_helpers import get_df_for_scores
from data_utils import get_mouse_v1_data

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
print(f"{DATA_PATH=}")


config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
    },
    "device": os.environ["DEVICE"],
    "crop_win": (22, 36),
    "seed": 0,
    "save_path": os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
    "load_ckpt": None,
    # "load_ckpt": os.path.join(DATA_PATH, "models", "encoder_sens22_mall_mean_activity.pth"),
    "only_eval": False,
}

### data config
config["data"]["mouse_v1"] = {
    "dataset_fn": "sensorium.datasets.static_loaders",
    "dataset_config": {
        "paths": [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
            # os.path.join(DATA_PATH, "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # mouse 1
            # os.path.join(DATA_PATH, "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # sensorium+ (mouse 2)
            os.path.join(DATA_PATH, "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 3)
            os.path.join(DATA_PATH, "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 4)
            os.path.join(DATA_PATH, "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 5)
            os.path.join(DATA_PATH, "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 6)
            os.path.join(DATA_PATH, "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"), # pretraining (mouse 7)
        ],
        "normalize": True,
        "scale": 0.25, # 256x144 -> 64x36
        "include_behavior": False,
        "add_behavior_as_channels": False,
        "include_eye_position": True,
        "exclude": None,
        "file_tree": True,
        "cuda": "cuda" in config["device"],
        "batch_size": 32,
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
_dataloaders, _ = get_mouse_v1_data(config=config["data"])
config["model_fn"] = "sensorium.models.stacked_core_full_gauss_readout"
config["model_config"] = {
    "pad_input": False,
    "layers": 4,
    "input_kern": 9,
    "gamma_input": 6.3831,
    "gamma_readout": 0.0076,
    "hidden_kern": 7,
    "hidden_channels": 64,
    "depth_separable": True,
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
    "shifter": True,
    "stack": -1,
    "mean_activity_dict": {
        data_key: torch.from_numpy(dat.statistics["responses"]["all"]["mean"]).to(config["device"])
        for data_key, dat in zip(_dataloaders["mouse_v1"]["train"].data_keys, _dataloaders["mouse_v1"]["train"].datasets)
    },
}
del _dataloaders

### trainer config
config["trainer_fn"] = "sensorium.training.standard_trainer"
config["trainer_config"] = {
    "max_iter": 200,
    "verbose": True,
    "lr_decay_steps": 4,
    "avg_loss": False,
    "lr_init": 0.009,
    "track_training": True,
}


class Neurons:
    def __init__(self, coords):
        self.cell_motor_coordinates = coords

if __name__ == "__main__":
    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders, neuron_coords = get_mouse_v1_data(config=config["data"])
    data_keys = _dataloaders["mouse_v1"]["train"].data_keys
    dataloaders = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["test"].dataloaders)}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=config["model_fn"],
        model_config=config["model_config"],
        dataloaders=dataloaders,
        seed=config["seed"],
    )

    ### load ckpt
    if config["load_ckpt"] is not None:
        print(f"[INFO] Loading ckpt from {config['load_ckpt']}")
        model.load_state_dict(torch.load(config["load_ckpt"], pickle_module=dill)["model"])
    model.to(config["device"])
    print(f"[INFO] Config: {json.dumps(config, indent=2, default=str)}")

    if not config["only_eval"]:
        ### train
        trainer = get_trainer(trainer_fn=config["trainer_fn"], trainer_config=config["trainer_config"])
        print(f"[INFO] Training starts...")
        validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=config["seed"])
        print(f"{trainer_output=}")
        print(f"{validation_score=}")

        ### save
        print(f"[INFO] Saving the model...")
        torch.save({
                "config": config,
                "model": model.state_dict(),
                "val_score": validation_score,
                "trainer_output": trainer_output,
                "state_dict": state_dict,
            }, config["save_path"], pickle_module=dill,
        )

    print(f"[INFO] Evaluating single trial correlation...")
    model.eval()
    single_trial_correlation = get_correlations(model, dataloaders, tier="test", device=config["device"], as_dict=True)
    df_corr_mean = get_df_for_scores(session_dict=single_trial_correlation, measure_attribute="Single Trial Correlation").groupby("dataset").mean()
    print(df_corr_mean)

    print(f"[INFO] Evaluating correlation to average...")
    dataloaders_for_eval = get_data(config["data"]["mouse_v1"]["dataset_fn"], config["data"]["mouse_v1"]["dataset_config"])
    correlation_to_average = get_signal_correlations(model, dataloaders_for_eval, tier="test", device=config["device"], as_dict=True)
    df_corr_avr = get_df_for_scores(session_dict=correlation_to_average, measure_attribute="Correlation to Average").groupby("dataset").mean()
    print(df_corr_avr)

    print(f"[INFO] Evaluating validation and test loss...")
    print("Validation loss: ", get_poisson_loss(
        model,
        dataloaders["validation"],
        device=config["device"],
        as_dict=False,
        avg=True,
        per_neuron=False,
        eps=1e-12,
    ))
    print("Test loss: ", get_poisson_loss(
        model,
        dataloaders["test"],
        device=config["device"],
        as_dict=False,
        avg=True,
        per_neuron=False,
        eps=1e-12,
    ))

import os
import numpy as np
import json
import dill
from pathlib import Path
from copy import deepcopy
import torch
from collections import OrderedDict, namedtuple
from nnfabrik.builder import get_model, get_trainer
from sensorium.utility import get_correlations, get_signal_correlations
from sensorium.utility.scores import get_poisson_loss
from sensorium.utility.measure_helpers import get_df_for_scores

from csng.utils import seed_all
from data import get_brainreader_data

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "brainreader")
print(f"{DATA_PATH=}")


config = {
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
    },
    "device": os.environ["DEVICE"],
    "seed": 4,
    "save_path": os.path.join(DATA_PATH, "models", "encoder_m6_seed4.pth"),
    "load_ckpt": None,
    # "load_ckpt": os.path.join(DATA_PATH, "models", "encoder_mall.pth"),
    "only_eval": False,
}

### data config
config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH, "data"),
    "batch_size": 64,
    # "sessions": list(range(1, 23)),
    "sessions": [6],
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}

### model
_dls = get_brainreader_data(config=config["data"]["brainreader_mouse"])
config["model_fn"] = "sensorium.models.stacked_core_full_gauss_readout"
config["model_config"] = {
    "pad_input": False,
    "layers": 3,
    "input_kern": 15,
    "gamma_input": 6.3831,
    "gamma_readout": 0.0076,
    "hidden_kern": 7,
    "hidden_channels": 32,
    "hidden_padding": 3,
    "depth_separable": False,
    # "laplace_pyramid": True, # TODO: add from brainreader.encoder_models.py
    # "grid_mean_predictor": {
    #     "type": "cortex",
    #     "input_dimensions": 2,
    #     "hidden_layers": 1,
    #     "hidden_features": 30,
    #     "final_tanh": True,
    # },
    "grid_mean_predictor": None,
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
    "shifter": False,
    "stack": -1,
    "mean_activity_dict": {
        data_key: torch.from_numpy(np.load(os.path.join(Path(dset.dataset_dir).parent.absolute(), "stats", "responses_mean.npy"))).to(config["device"])
        for data_key, dset in _dls["brainreader_mouse"]["train"].datasets.items()
    },
}
del _dls

### trainer config
config["trainer_fn"] = "sensorium.training.standard_trainer"
config["trainer_config"] = {
    "max_iter": 200,
    "verbose": True,
    "lr_decay_steps": 4,
    "avg_loss": False,
    "lr_init": 0.03,
    "track_training": True,
    "weight_decay": 0.,
}


# class Neurons:
#     def __init__(self, n_neurons, coords=None):
#         self.unit_ids = np.arange(n_neurons)
#         self.cell_motor_coordinates = coords


if __name__ == "__main__":
    seed_all(config["seed"])

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = get_brainreader_data(config=config["data"]["brainreader_mouse"])
    data_keys = _dataloaders["brainreader_mouse"]["train"].data_keys
    dataloaders = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["test"].dataloaders)}),
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
        }, config["save_path"], pickle_module=dill)

    model.eval()

    # print(f"[INFO] Evaluating single trial correlation...")
    print(f"[INFO] Evaluating correlation to average...") # responses averaged across trials already
    correlation_to_avg = get_correlations(model, dataloaders, tier="test", device=config["device"], as_dict=True)
    df_corr_avg = get_df_for_scores(session_dict=correlation_to_avg, measure_attribute="Correlation to Average").groupby("dataset").mean()
    print(df_corr_avg)

    # config_for_eval = deepcopy(config)
    # config_for_eval["data"]["brainreader_mouse"]["avg_test_resp"] = False
    # _dataloaders_for_eval = get_brainreader_data(config=config_for_eval["data"]["brainreader_mouse"])
    # _data_keys = _dataloaders_for_eval["brainreader_mouse"]["train"].data_keys
    # dataloaders_for_eval = OrderedDict({
    #     "train": OrderedDict({data_key: dl for data_key, dl in zip(_data_keys, _dataloaders_for_eval["brainreader_mouse"]["train"].dataloaders)}),
    #     "validation": OrderedDict({data_key: dl for data_key, dl in zip(_data_keys, _dataloaders_for_eval["brainreader_mouse"]["val"].dataloaders)}),
    #     "test": OrderedDict({data_key: dl for data_key, dl in zip(_data_keys, _dataloaders_for_eval["brainreader_mouse"]["test"].dataloaders)}),
    # })
    # for data_key, dl in dataloaders_for_eval["test"].items():
    #     dset = dl.dataset
    #     dset.neurons = Neurons(n_neurons=dset[0].responses.shape[-1])
    #     dset.trial_info = namedtuple("trial_info", ["tiers", "frame_image_id", "trial_idx"])(
    #         ["test"]*len(dset), list(range(len(dset))), list(range(len(dset)))
    #     )
    # correlation_to_average = get_signal_correlations(model, dataloaders_for_eval, tier="test", device=config_for_eval["device"], as_dict=True)
    # df_corr_avg = get_df_for_scores(session_dict=correlation_to_average, measure_attribute="Correlation to Average").groupby("dataset").mean()
    # print(df_corr_avg)

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

import os
import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from csng.utils.mix import update_config_paths
from csng.cat_v1.data import get_cat_v1_dataloaders

DATA_PATH_CAT_V1 = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")


class Neurons:
    def __init__(self, coords):
        self.cell_motor_coordinates = coords

def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill, map_location=device)
    ckpt["config"] = update_config_paths(config=ckpt["config"], new_data_path=DATA_PATH_CAT_V1, replace_until_folder="50K_single_trial_dataset")
    ckpt["config"]["data"]["cat_v1"]["dataset_config"]["device"] = device

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = get_cat_v1_dataloaders(**ckpt["config"]["data"]["cat_v1"]["dataset_config"])
    for k in _dataloaders.keys():
        _dataloaders[k].dataset.neurons = Neurons(coords=_dataloaders[k].dataset.coords["all"].numpy())
    data_keys = ["cat_v1"]
    dataloaders = OrderedDict({
        "train": OrderedDict({"cat_v1": _dataloaders["train"]}),
        "validation": OrderedDict({"cat_v1": _dataloaders["val"]}),
        "test": OrderedDict({"cat_v1": _dataloaders["test"]}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=ckpt["config"]["model_fn"],
        model_config=ckpt["config"]["model_config"],
        dataloaders=dataloaders,
        seed=ckpt["config"]["seed"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    if eval_mode:
        model.eval()

    return model

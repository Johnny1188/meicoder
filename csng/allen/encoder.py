import os
import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from csng.utils.mix import update_config_paths
from csng.allen.data import get_allen_dataloaders

DATA_PATH_CAE = os.path.join(os.environ["DATA_PATH"], "cae")


def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill, map_location=device)
    ckpt["config"] = update_config_paths(config=ckpt["config"], new_data_path=DATA_PATH_CAE, replace_until_folder="cae")
    ckpt["config"]["data"]["allen"]["device"] = device

    ### prepare dataloaders compatible w/ nnfabrik
    _dls = get_allen_dataloaders(config=ckpt["config"]["data"]["allen"])["allen"]
    dls = OrderedDict({
        "train": OrderedDict({"allen": _dls["train"]}),
        "validation": OrderedDict({"allen": _dls["val"]}),
        "test": OrderedDict({"allen": _dls["test"]}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=ckpt["config"]["model_fn"],
        model_config=ckpt["config"]["model_config"],
        dataloaders=dls,
        seed=ckpt["config"]["seed"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    if eval_mode:
        model.eval()

    return model

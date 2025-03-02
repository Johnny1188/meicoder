import os
import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from csng.utils.mix import update_config_paths
from csng.mouse_v1.data import get_mouse_v1_dataloaders

DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")


def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill, map_location=device)
    ckpt["config"] = update_config_paths(config=ckpt["config"], new_data_path=DATA_PATH_MOUSE_V1)
    ckpt["config"]["data"]["mouse_v1"]["device"] = device

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = get_mouse_v1_dataloaders(config=ckpt["config"])[0]["mouse_v1"]
    data_keys = _dataloaders["train"].data_keys
    dataloaders = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["test"].dataloaders)}),
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

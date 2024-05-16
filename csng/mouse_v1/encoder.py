import os
import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from data_utils import get_mouse_v1_data

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill)

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders, neuron_coords = get_mouse_v1_data(config=ckpt["config"]["data"])
    data_keys = _dataloaders["mouse_v1"]["train"].data_keys
    dataloaders = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["mouse_v1"]["test"].dataloaders)}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=ckpt["config"]["model_fn"],
        model_config=ckpt["config"]["model_config"],
        dataloaders=dataloaders,
        seed=ckpt["config"]["seed"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    if eval_mode:
        model.eval()

    return model

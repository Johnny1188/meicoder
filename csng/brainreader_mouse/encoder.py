import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from csng.brainreader_mouse.data import get_brainreader_data


def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill)

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = get_brainreader_data(config=ckpt["config"]["data"]["brainreader_mouse"])
    data_keys = _dataloaders["brainreader_mouse"]["train"].data_keys
    dataloaders = OrderedDict({
        "train": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["train"].dataloaders)}),
        "validation": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["val"].dataloaders)}),
        "test": OrderedDict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["test"].dataloaders)}),
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

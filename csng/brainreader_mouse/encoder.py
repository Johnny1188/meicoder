import os
import torch
import dill
from collections import OrderedDict
from nnfabrik.builder import get_model
from csng.utils.mix import update_config_paths
from csng.brainreader_mouse.data import get_brainreader_mouse_dataloaders

DATA_PATH_BRAINREADER = os.path.join(os.environ["DATA_PATH"], "brainreader")


def get_encoder(ckpt_path, eval_mode=True, device="cpu"):
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill, map_location=device)
    ckpt["config"] = update_config_paths(config=ckpt["config"], new_data_path=DATA_PATH_BRAINREADER, replace_until_folder="brainreader")
    ckpt["config"]["data"]["brainreader_mouse"]["device"] = device

    ### prepare dataloaders compatible w/ nnfabrik
    _dataloaders = get_brainreader_mouse_dataloaders(config=ckpt["config"]["data"]["brainreader_mouse"])["brainreader_mouse"]
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

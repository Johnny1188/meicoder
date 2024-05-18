import os
import dill
import torch
from collections import OrderedDict
from nnfabrik.builder import get_model

from data import prepare_v1_dataloaders

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "cat_V1_spiking_model", "50K_single_trial_dataset")


def get_encoder(device="cpu", eval_mode=True, ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = os.path.join(DATA_PATH, "models", "encoder_cat_v1_no_shifter.pth")
    print(f"[INFO] Loading encoder checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, pickle_module=dill)

    ### prepare dataloaders compatible w/ nnfabrik
    class Neurons:
        def __init__(self, coords):
            self.cell_motor_coordinates = coords
    _dataloaders = prepare_v1_dataloaders(**ckpt["config"]["data"]["cat_v1"])
    for k in _dataloaders.keys():
        _dataloaders[k].dataset.neurons = Neurons(
            coords=_dataloaders[k].dataset.coords["all"].numpy()
        )
    dataloaders = OrderedDict({
        "train": OrderedDict({"cat_v1": _dataloaders["train"]}),
        "validation": OrderedDict({"cat_v1": _dataloaders["val"]}),
        "test": OrderedDict({"cat_v1": _dataloaders["test"]}),
    })

    ### build the encoder model
    model = get_model(
        model_fn=ckpt["model_fn"],
        model_config=ckpt["model_config"],
        dataloaders=dataloaders,
        seed=ckpt["config"]["seed"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    if eval_mode:
        model.eval()

    return model

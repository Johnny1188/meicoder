import os
import torch
from torchvision import transforms

from tqdm import tqdm
from data import get_brainreader_mouse_dataloaders

DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")
device = os.environ["DEVICE"]

config = {
    "device": os.environ["DEVICE"],
    "seed": 0,
    "data": {
        "mixing_strategy": "sequential", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
    "save_path": os.path.join(DATA_PATH, "models", "imagenet_decoder.pt"),
    "train": True,
}

config["data"]["brainreader_mouse"] = {
    "device": config["device"],
    "mixing_strategy": config["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join(DATA_PATH_BRAINREADER, "data"),
    "batch_size": 128,
    "sessions": list(range(1, 23)),
    # "sessions": [6],
    # "resize_stim_to": (36, 64),
    "normalize_stim": True,
    "normalize_resp": False,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}

data_loader = get_brainreader_mouse_dataloaders(config['data']['brainreader_mouse'])["brainreader_mouse"]

dl = data_loader["train"]
print("Batches: ", len(dl))
batch = next(dl)
print("Batch length: ", len(batch))
print(batch[0].keys())

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
resnet50.eval().to(device)


for batch_no, batch in tqdm(enumerate(dl)):
    resnet50.register_hook()
    def objective(real, fake):
        pass


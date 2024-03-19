import os
import torch

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


def get_encoder(device="cpu", eval_mode=True, use_shifter=True, ckpt_path=None):
    ### get dataloaders
    filenames = [ # from https://gin.g-node.org/cajal/Sensorium2022/src/master
        "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # mouse 1
        "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # sensorium+ (mouse 2)
        "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 3)
        "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 4)
        "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 5)
        "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 6)
        "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip", # pretraining (mouse 7)
    ]
    for f_idx, f_name in enumerate(filenames):
        filenames[f_idx] = os.path.join(DATA_PATH, f_name)
    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {
        'paths': filenames,
        'normalize': True,
        'include_behavior': False,
        'include_eye_position': True,
        'batch_size': 64,
        'scale':.25,

        "use_cache": False,
    }
    dataloaders = get_data(dataset_fn, dataset_config)

    ### get model
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {
        'pad_input': False,
        'layers': 4,
        'input_kern': 9,
        'gamma_input': 6.3831,
        'gamma_readout': 0.0076,
        'hidden_kern': 7,
        'hidden_channels': 64,
        'depth_separable': True,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True
        },
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'shifter': use_shifter,
        'stack': -1,
    }
    model = get_model(
        model_fn=model_fn,
        model_config=model_config,
        dataloaders=dataloaders,
        seed=42,
    ).to(device)
    if ckpt_path is None:
        ckpt_path = os.path.join(DATA_PATH, "models", "encoder_sens22.pth")
    print(f"Loading encoder checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    if eval_mode:
        model.eval()

    del dataloaders
    return model

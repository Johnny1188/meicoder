import os
import torch
import numpy as np
import dill
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from csng.utils.mix import update_config_paths, seed_all
from csng.data import get_dataloaders


class DenseDecoder(nn.Module):
    """
    Decodes a 1D spike vector into a 2D latent representation.
    Corresponds to the Keras 'dense_decoder'.
    """
    def __init__(self, ncell, size="large", intermediate_shape=(1, 64, 64)):
        super().__init__()
        if size == "large":
            self.decoder = nn.Sequential(
                nn.Linear(ncell, 16384),
                nn.BatchNorm1d(16384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                
                nn.Linear(16384, 8192),
                nn.BatchNorm1d(8192),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                
                nn.Linear(8192, np.prod(intermediate_shape)),
                nn.Sigmoid()
            )
        elif size == "small":
            self.decoder = nn.Sequential(
                nn.Linear(ncell, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                
                nn.Linear(4096, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),

                nn.Linear(2048, np.prod(intermediate_shape)),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown size '{size}' for DenseDecoder. Use 'large' or 'small'.")

    def forward(self, x):
        return self.decoder(x)


class CAE(nn.Module):
    """
    The original asymmetric Convolutional Autoencoder from the Keras code.
    It takes a (64, 64) input and produces a (256, 256) output.
    """
    def __init__(self, input_channels=1, size="large"):
        super().__init__()
        if size == "large":
            layers = [
                # --- Encoder (4 down-sampling steps) ---
                # Input: (N, 1, 64, 64)
                nn.Conv2d(input_channels, 256, kernel_size=7, stride=2, padding=3), # -> (N, 256, 32, 32)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2), # -> (N, 512, 16, 16)
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # -> (N, 1024, 8, 8)
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # -> (N, 1024, 4, 4)
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
            ]
            layers.extend([
                # --- Decoder (6 up-sampling steps) ---
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 1024, 8, 8)
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 1024, 16, 16)
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
        
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 512, 32, 32)
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
        
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 256, 64, 64)
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 256, 128, 128)
                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 128, 256, 256)
                nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3), # Final layer
                nn.BatchNorm2d(1) # Corresponds to `BatchNormalization(name = 'cae_out')`
            ])
        elif size == "small":
            layers = [
                # --- Encoder (4 down-sampling steps) ---
                # Input: (N, 1, 64, 64)
                nn.Conv2d(input_channels, 128, kernel_size=7, stride=2, padding=3), # -> (N, 128, 32, 32)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # -> (N, 256, 16, 16)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1), # -> (N, 320, 8, 8)
                nn.BatchNorm2d(320),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1), # -> (N, 320, 4, 4)
                nn.BatchNorm2d(320),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
            ]
            layers.extend([
                # --- Decoder (3 up-sampling steps) ---
                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 320, 8, 8)
                nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(320),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 320, 16, 16)
                nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),

                nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 128, 32, 32)
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
        
                # nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 128, 256, 256)
                nn.Conv2d(128, 1, kernel_size=5, stride=1, padding=2), # Final layer
                nn.BatchNorm2d(1) # Corresponds to `BatchNormalization(name = 'cae_out')`
            ])
        else:
            raise ValueError(f"Unknown size '{size}' for CAE. Use 'large' or 'small'.")

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class End2EndModel(nn.Module):
    """
    The full end-to-end model that connects the DenseDecoder and the CAE.
    """
    def __init__(self, ncell, size, intermediate_shape):
        super().__init__()
        self.intermediate_shape = intermediate_shape
        self.dense_decoder = DenseDecoder(ncell, size=size, intermediate_shape=intermediate_shape)
        self.cae = CAE(input_channels=1, size=size)

    def forward(self, x, return_intermediate=False):
        # Pass through the dense decoder
        dense_out_flat = self.dense_decoder(x.squeeze(-1))
        
        # Reshape to image format (N, C, H, W) for the CAE
        dense_out_reshaped = dense_out_flat.view(-1, *self.intermediate_shape)

        # Pass through the convolutional autoencoder
        cae_out = self.cae(dense_out_reshaped)

        if return_intermediate:
            return dense_out_reshaped, cae_out
        else:
            return cae_out


class CAEDecoder(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        ### initialize model and load checkpoint
        self.cfg = kwargs
        self.ckpt_path = None
        if "ckpt_path" in self.cfg:
            self.ckpt_path = self.cfg["ckpt_path"]
            del self.cfg["ckpt_path"]
            ckpt = torch.load(self.ckpt_path, pickle_module=dill)
            self.cfg.update(ckpt["config"]["model"]["kwargs"])
        self.model = End2EndModel(**self.cfg)

        if self.ckpt_path is not None:
            ckpt["model"] = {k[:6].replace("model.", "") + k[6:]: v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(ckpt["model"])

    def forward(self, x, data_key=None, neuron_coords=None, pupil_center=None):
        recons = self.model(x, return_intermediate=False)
        return recons

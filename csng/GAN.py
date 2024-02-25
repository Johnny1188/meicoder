import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        in_shape=(1, 51, 51),
        layers=[("deconv", 8, 60, 1, 0), ("conv", 16, 5, 1, "same"), ("conv", 1, 3, 1, 0)],
        act_fn=nn.ReLU,
        out_act_fn=nn.Identity,
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.layers = layers
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        self._build_layers()

    def _build_layers(self):
        layers = []
        in_channels = self.in_shape[0]
        
        ### build cnn layers
        for l_i, layer_config in enumerate(self.layers):
            if layer_config[0] == "fc":
                layer_type, out_channels = layer_config
                layers.append(nn.Linear(in_channels, out_channels))
            elif layer_config[0] == "unflatten":
                layer_type, in_dim, unflattened_size = layer_config
                layers.append(nn.Unflatten(in_dim, unflattened_size))
                out_channels = unflattened_size[0]
            elif layer_config[0] == "maxpool":
                layer_type, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                )
                out_channels = in_channels
            elif layer_config[0] == "upsample":
                layer_type, scale_factor = layer_config
                layers.append(nn.Upsample(scale_factor=scale_factor))
                out_channels = in_channels
            elif layer_config[0] == "deconv":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            elif layer_config[0] == "conv":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            else:
                raise ValueError(f"layer_type {layer_type} not recognized")

            if l_i < len(self.layers) - 1 and layer_type in ["fc", "conv", "deconv"]:
                ### add batch norm, activation, dropout
                if self.batch_norm:
                    if layer_type == "fc":
                        layers.append(nn.BatchNorm1d(out_channels))
                    else:
                        layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.act_fn())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            elif l_i == len(self.layers) - 1:
                ### add output activation
                layers.append(self.out_act_fn())

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_shape=(1, 110, 110),
        layers=[("conv", 32, 5, 2, 1), ("conv", 32, 5, 2, 1), ("conv", 16, 3, 1, 0), ("fc", 1)],
        act_fn=nn.LeakyReLU,
        out_act_fn=nn.Sigmoid,
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.layers = layers
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        self._build_layers()

    def _get_out_shape(self, layers):
        x = torch.zeros(1, *self.in_shape) # dummy input
        for l in layers:
            x = l(x)
        return x.shape[1:]

    def _build_layers(self):
        layers = []
        in_channels = self.in_shape[0]

        ### build cnn layers
        for l_i, layer_config in enumerate(self.layers):
            if layer_config[0] == "conv":
                layer_type, out_channels, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            elif layer_config[0] == "maxpool":
                layer_type, kernel_size, stride, padding = layer_config
                layers.append(
                    nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                )
                out_channels = in_channels
            elif layer_config[0] == "fc":
                layer_type, out_channels = layer_config
                layers.append(nn.Flatten())
                layers.append(nn.Linear(np.prod(self._get_out_shape(layers)), out_channels))
            else:
                raise ValueError(f"layer_type {layer_type} not recognized")

            if l_i < len(self.layers) - 1 and layer_type in ["fc", "conv"]:
                ### add batch norm, activation, dropout
                if self.batch_norm:
                    if layer_type == "fc":
                        layers.append(nn.BatchNorm1d(out_channels))
                    else:
                        layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.act_fn())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            elif l_i == len(self.layers) - 1:
                ### add output activation
                layers.append(self.out_act_fn())

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class GAN(nn.Module):
    def __init__(
        self,
        G_kwargs,
        D_kwargs,
        G_optim_kwargs={"lr": 1e-4, "betas": (0.5, 0.999)},
        D_optim_kwargs={"lr": 1e-4, "betas": (0.5, 0.999)},
    ):
        super().__init__()
        self.G_kwargs = G_kwargs
        self.D_kwargs = D_kwargs
        self.G_optim_kwargs = G_optim_kwargs
        self.D_optim_kwargs = D_optim_kwargs

        self._build_models()
        self._build_optimizers()

    def _build_models(self):
        self.G = Generator(**self.G_kwargs)
        self.D = Discriminator(**self.D_kwargs)

    def _build_optimizers(self):
        self.G_optim = torch.optim.Adam(self.G.parameters(), **self.G_optim_kwargs)
        self.D_optim = torch.optim.Adam(self.D.parameters(), **self.D_optim_kwargs)

    def state_dict(self, destination=None, prefix=None, keep_vars=True):
        state_dict = {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "G_optim": self.G_optim.state_dict(),
            "D_optim": self.D_optim.state_dict(),
        }

        if destination is not None:
            for k, v in state_dict.items():
                prefix = prefix if prefix is not None else ""
                destination[prefix + k] = v
            return destination

        return state_dict

    def load_state_dict(self, state_dict):
        self.G.load_state_dict(state_dict["G"])
        self.D.load_state_dict(state_dict["D"])
        self.G_optim.load_state_dict(state_dict["G_optim"])
        self.D_optim.load_state_dict(state_dict["D_optim"])

    def forward(self, x):
        return self.G(x)

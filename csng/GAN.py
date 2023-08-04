import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        img_shape=(1, 51, 51),
        layers=[("deconv", 8, 60, 1, 0), ("conv", 16, 5, 1, "same"), ("conv", 1, 3, 1, 0)],
        act_fn=nn.ReLU,
        out_act_fn=nn.Identity,
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.layers = layers
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        self._build_layers()

    def _build_layers(self):
        layers = []
        in_channels = self.img_shape[0]
        
        ### build cnn layers
        for l_i, layer_config in enumerate(self.layers):
            if layer_config[0] == "deconv":
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

            if l_i < len(self.layers) - 1:
                ### add batch norm, activation, dropout
                if self.batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.act_fn())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            else:
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
            elif layer_config[0] == "fc":
                layer_type, out_channels = layer_config
                layers.append(nn.Flatten())
                layers.append(nn.Linear(np.prod(self._get_out_shape(layers)), out_channels))
            else:
                raise ValueError(f"layer_type {layer_type} not recognized")

            if l_i < len(self.layers) - 1:
                ### add batch norm, activation, dropout
                if self.batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.act_fn())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            else:
                ### add output activation
                layers.append(self.out_act_fn())

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class GAN(nn.Module):
    def __init__(
        self,
        gen_kwargs,
        disc_kwargs,
        gen_optim_kwargs={"lr": 1e-4, "betas": (0.5, 0.999)},
        disc_optim_kwargs={"lr": 1e-4, "betas": (0.5, 0.999)},
    ):
        super().__init__()
        self.gen_kwargs = gen_kwargs
        self.disc_kwargs = disc_kwargs
        self.gen_optim_kwargs = gen_optim_kwargs
        self.disc_optim_kwargs = disc_optim_kwargs

        self._build_models()
        self._build_optimizers()

    def _build_models(self):
        self.gen = Generator(**self.gen_kwargs)
        self.disc = Discriminator(**self.disc_kwargs)

    def _build_optimizers(self):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), **self.gen_optim_kwargs)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), **self.disc_optim_kwargs)

    def forward(self, x):
        return self.gen(x)

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CNN_Decoder(nn.Module):
    def __init__(
        self,
        resp_shape=(10000,),
        stim_shape=(1, 110, 110),
        layers=[("deconv", 1, 60, 1, 0), ("conv", 1, 3, 1, 1)],
        act_fn=nn.ReLU,
        out_act_fn=nn.Identity,
        dropout=0.0,
        batch_norm=False,
    ):
        # assert layers[-1][0] == resp_shape[0], "last layer should have same number of channels as resp_shape[0]"
        super().__init__()
        self.resp_shape = resp_shape
        self.stim_shape = stim_shape
        self.layers = layers
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        self._build_layers()

    def _build_layers(self):
        layers = []
        in_channels = self.resp_shape[0]
        
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

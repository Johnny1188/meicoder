import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    """Convolutional Neural Network (CNN) model.

    :param in_shape: input shape of the data
    :param layers: list of tuples, each tuple specifies a layer in the model
    :param act_fn: activation function to use
    :param out_act_fn: output activation function to use
    :param dropout: dropout rate
    :param batch_norm: whether to use batch normalization
    """

    def __init__(
        self,
        in_shape=(480,),
        layers=[("deconv", 1, 60, 1, 0), ("conv", 1, 3, 1, 1)],
        act_fn=nn.ReLU,
        out_act_fn=nn.Identity,
        dropout=0.35,
        batch_norm=True,
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

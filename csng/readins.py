import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from torchvision.transforms.functional import gaussian_blur

from csng.GAN import GAN
from csng.CNN_Decoder import CNN_Decoder
from csng.utils import build_layers


class MultiReadIn(nn.Module):
    def __init__(self, readins_config, core_cls, core_config, crop_stim_fn=None):
        super().__init__()

        ### init all readin layers
        self.readins_config = readins_config # 1 ReadIn <=> 1 FC layer
        self.readins = nn.ModuleDict()
        for readin_config in readins_config:
            ### init readin layer
            in_channels = readin_config["in_shape"]
            readin = []
            for layer_config in readin_config["layers"]:
                if type(layer_config[0]) == type:
                    layer_cls, _layer_config = layer_config[0], layer_config[1]
                    readin.append(layer_cls(**_layer_config))
                    if "decoding_objective_config" in readin_config \
                         and readin_config["decoding_objective_config"] is not None:
                        readin[-1].setup_decoding_objective(
                            **readin_config["decoding_objective_config"]
                        )
                    if hasattr(readin[-1], "out_channels"):
                        in_channels = readin[-1].out_channels
                    else:
                        in_channels = layer_config[2]
                else:
                    raise ValueError(f"layer_type {layer_config[0]} not recognized")

            self.readins[readin_config["data_key"]] = nn.ModuleList(readin)

        ### core decoder
        self.core_config = deepcopy(core_config)
        if "resp_shape" in self.core_config:
            print(f"[WARNING] resp_shape in core_config will be overwritten by the output of the last readin layer (shape {in_channels})")
            self.core_config["resp_shape"] = (in_channels,)
        self.core = core_cls(**self.core_config)

        ### crop stim
        self.crop_stim_fn = crop_stim_fn

    def set_additional_loss(self, inp, out):
        for readin_data_key, readin in self.readins.items():
            if "data_key" in inp and inp["data_key"] != readin_data_key:
                continue
            for l in readin:
                out["encoded"] = l(inp["resp"], neuron_coords=inp["neuron_coords"] if "neuron_coords" in inp else None,
                                   pupil_center=inp["pupil_center"] if "pupil_center" in inp else None)
                if hasattr(l, "set_additional_loss"):
                    l.set_additional_loss(inp, out)

    def get_additional_loss(self, data_key=None):
        loss = 0.
        for readin_data_key, readin in self.readins.items():
            if data_key is not None and readin_data_key != data_key:
                continue
            for l in readin:
                if hasattr(l, "get_additional_loss"):
                    loss += l.get_additional_loss()
        return loss

    def forward(self, x, data_key=None, neuron_coords=None, pupil_center=None, additional_core_inp=None):
        assert data_key is not None or len(self.readins) == 0, \
            "data_key must be provided if there are multiple readin layers"

        ### run readin layer
        if data_key is not None:
            for l in self.readins[data_key]:
                x = l(x, neuron_coords=neuron_coords, pupil_center=pupil_center)

        ### run core decoder
        x = self.core(x) if additional_core_inp is None else self.core(x, **additional_core_inp)
        if self.crop_stim_fn:
            x = self.crop_stim_fn(x)

        return x


class ReadIn(nn.Module):
    """
    Takes in:
        - neuronal responses (B, N_neurons),
        - neuronal coordinates (N_neurons, 3),
        - pupil center (B, 2).
    """
    def __init__(self):
        super().__init__()
        self._last_loss = 0.

    def setup_decoding_objective(
        self,
        decoder_cls,
        decoder_config={
            "in_shape": 384,
            "layers_config": [("fc", 400), ("fc", 7334)],
            "act_fn": nn.LeakyReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.1,
            "batch_norm": False,
        },
        loss_fn=nn.MSELoss(),
    ):
        self.decoder = decoder_cls(**decoder_config)
        self.decoding_loss_fn = loss_fn

    def set_additional_loss(self, inp, out):
        if hasattr(self, "decoding_loss_fn"):
            self._last_loss = self.decoding_loss_fn(
                self.decoder(out["encoded"].view(inp["resp"].size(0), -1)),
                inp["resp"]
            )

    def get_additional_loss(self):
        return self._last_loss

    def forward(self, x, neuron_coords=None, pupil_center=None):
        raise NotImplementedError


class FCReadIn(ReadIn):
    def __init__(
        self,
        in_shape,
        layers_config=[("fc", 20), ("fc", 2)],
        act_fn=nn.LeakyReLU,
        out_act_fn=nn.Identity,
        dropout=0.0,
        batch_norm=False,
        out_channels=None,
    ):
        super().__init__()
        self.requires_neuron_coords = False
        self.requires_pupil_center = False

        self.layers = build_layers(
            in_channels=in_shape,
            layers_config=layers_config,
            act_fn=act_fn,
            out_act_fn=out_act_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = layers_config[-1][-1]

    def forward(self, x, neuron_coords=None, pupil_center=None):
        return self.layers(x)


class AutoEncoderReadIn(ReadIn):
    def __init__(
        self,
        encoder_config,
        decoder_config={
            "layers_config": [("fc", 20), ("fc", 2)],
            "act_fn": nn.LeakyReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.0,
            "batch_norm": False,
            "out_channels": None,
        },
        loss_mul=1.0,
    ):
        super().__init__()
        self.requires_neuron_coords = False
        self.requires_pupil_center = False

        self.encoder = FCReadIn(**encoder_config)
        self.decoder = FCReadIn(
            in_shape=encoder_config["layers_config"][-1][-1] \
                if type(encoder_config["layers_config"][-1][-1]) == int \
                else np.prod(encoder_config["layers_config"][-1][-1]),
            **decoder_config,
        )

        self.out_channels = self.encoder.out_channels

        self.loss_mul = loss_mul
        self._last_loss = 0.

    def set_additional_loss(self, orig, encoded):
        self._last_loss = self.loss_mul * F.mse_loss(self.decoder(encoded), orig)

    def get_additional_loss(self):
        return self._last_loss

    def forward(self, x, neuron_coords=None, pupil_center=None):
        acts = self.encoder(x)
        if self.training:
            self.set_additional_loss(orig=x, encoded=acts.view(x.size(0), -1))
        return acts


class ShifterNet(nn.Module):
    def __init__(
        self,
        in_channels,
        layers_config=[("fc", 20), ("fc", 2)],
        act_fn=nn.LeakyReLU,
        out_act_fn=nn.Tanh,
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()

        self.layers = build_layers(
            in_channels=in_channels,
            layers_config=layers_config,
            act_fn=act_fn,
            out_act_fn=out_act_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x):
        return self.layers(x)


class HypernetReadIn(ReadIn):
    """
    Takes in:
        - neuronal responses (B, N_neurons),
        - neuronal coordinates (N_neurons, 3),
        - pupil center (B, 2).
    Uses a hypernetwork to generate weights of the target layer.
    """
    def __init__(
        self,
        n_neurons,
        hypernet_layers=[("fc", 20), ("fc", 20), ("fc", 432)],
        hypernet_act_fn=nn.ReLU,
        hypernet_out_act_fn=nn.Identity,
        hypernet_dropout=0.0,
        hypernet_batch_norm=False,
        hypernet_init="xavier",
        hypernet_init_kwargs=None,
        hypernet_neuron_embed_dim=None, # learned neuron embeddings
        target_in_shape=10000,
        target_layers=[("fc", 432), ("unflatten", 1, (3, 9, 16))],
        target_act_fn=nn.ReLU,
        target_out_act_fn=nn.Identity,
        target_dropout=0.0,
        target_out_layer_norm=False,
        shift_coords=True,
        shifter_net_layers=[("fc", 20), ("fc", 2)],
        shifter_net_act_fn=nn.LeakyReLU,
        shifter_net_out_act_fn=nn.Tanh,
    ):
        assert target_layers[0][0] == "fc", "first layer of target must be fc"
        super().__init__()
        self.requires_neuron_coords = True
        self.requires_pupil_center = True

        self.hnet_config = {
            "in_channels": 3, # neuron_coords (x, y, z)
            "layers_config": hypernet_layers,
            "act_fn": hypernet_act_fn,
            "out_act_fn": hypernet_out_act_fn,
            "dropout": hypernet_dropout,
            "batch_norm": hypernet_batch_norm,
        }

        self.target_config = {
            "in_channels": target_in_shape,
            "layers_config": target_layers,
            "act_fn": target_act_fn() if type(target_act_fn) == type else target_act_fn,
            "out_act_fn": target_out_act_fn() if type(target_out_act_fn) == type else target_out_act_fn,
            "dropout": target_dropout,
            "layer_norm": target_out_layer_norm,
        }

        ### shifter net
        self.shift_coords = shift_coords
        self.shifter_net = None
        if shift_coords:
            self.shifter_net = ShifterNet( # pupil center (x, y) -> (delta_x, delta_y)
                in_channels=2,
                layers_config=shifter_net_layers,
                act_fn=shifter_net_act_fn,
                out_act_fn=shifter_net_out_act_fn,
                dropout=0.0,
                batch_norm=False,
            )

        self.hypernet_neuron_embed_dim = hypernet_neuron_embed_dim
        if self.hypernet_neuron_embed_dim:
            self.neuron_embed = nn.Embedding(
                num_embeddings=n_neurons,
                embedding_dim=hypernet_neuron_embed_dim,
            )
            self.hnet_config["in_channels"] += hypernet_neuron_embed_dim
        self.hypernet = build_layers(**self.hnet_config)
        self.target_bias = nn.Parameter(torch.zeros(target_layers[0][1]))
        self._init_hypernet_params(hypernet_init, hypernet_init_kwargs)


        self.out_channels = target_layers[-1][-1][0]

    def _init_hypernet_params(self, init, init_kwargs):
        ### init hypernet params
        for p in self.hypernet.parameters():
            if p.dim() > 1:
                if init == "normal":
                    if init_kwargs:
                        nn.init.normal_(p, **init_kwargs)
                    else:
                        nn.init.normal_(p)
                elif init == "xavier":
                    if init_kwargs:
                        nn.init.xavier_uniform_(p, **init_kwargs)
                    else:
                        nn.init.xavier_uniform_(p)
                else:
                    raise ValueError(f"init {init} not recognized")

    def forward(self, x, neuron_coords, pupil_center=None):
        ### expects neurons coords (n_neurons, 3)
        ### expects pupil_center (B, 2)

        B, n_neurons = x.shape

        ### shift neuron_coords by pupil_center
        neuron_coords = neuron_coords.unsqueeze(0).repeat(B, 1, 1) # (B, n_neurons, 3)
        if self.shift_coords:
            delta = self.shifter_net(pupil_center) # (B, 2)
            neuron_coords[:, torch.arange(n_neurons), :2] += delta.unsqueeze(1)

        ### get target layer params
        hypernet_inp = [neuron_coords]
        if self.hypernet_neuron_embed_dim:
            neuron_embed = self.neuron_embed(torch.arange(n_neurons, device=x.device))
            hypernet_inp.append(neuron_embed.unsqueeze(0).repeat(B, 1, 1))
        hypernet_inp = torch.cat(hypernet_inp, dim=-1) # (B, n_neurons, n_coords + n_neuron_embed_dim)
        hypernet_inp = hypernet_inp.view(B * n_neurons, -1) # (B * n_neurons, n_coords + n_neuron_embed_dim)
        target_params = self.hypernet(hypernet_inp) # (B * n_neurons, hidden_dim)
        
        ### split target_params to (B, n_neurons, n_target_params)
        target_params = target_params.view(B, n_neurons, target_params.shape[-1])

        ### run target layer
        for layer_config in self.target_config["layers_config"]:
            if layer_config[0] == "fc":
                ### W(B, n_neurons, n_target_params) @ x(B, n_neurons) + b(n_target_params)
                x = torch.einsum("bni,bn->bi", target_params, x) # sum over n_neurons
                x = x + self.target_bias

                ### layer norm
                if self.target_config["layer_norm"]:
                    x = F.layer_norm(x, x.shape[1:])
            elif layer_config[0] == "unflatten":
                _, in_dim, unflattened_size = layer_config
                x = torch.unflatten(x, dim=in_dim, sizes=unflattened_size)
            else:
                raise ValueError(f"layer_type {layer_config[0]} not recognized")

            if layer_config[0] in ["fc"]:
                ### add activation, dropout
                x = self.target_config["act_fn"](x)
                if self.target_config["dropout"] > 0.0:
                    x = F.dropout(x, p=self.target_config["dropout"])

        return x


class ConvReadIn(ReadIn):
    """
    Takes in:
        - neuronal responses (B, N_neurons),
        - neuronal coordinates (N_neurons, 3),
        - pupil center (B, 2).
    Reshapes the input according to the coordinates to a 3D tensor of shape (B, 3, H, W).
    Applies a convolutional layer to the 3D tensor.
    The pupil center is fed through a separate FC layer to get $Delta_x$ and $Delta_y$ that
    are added to the coordinates to shift the receptive field.
    """
    def __init__(
        self,
        H=9,
        W=16,
        shift_coords=True,
        shifter_net_layers=[("fc", 10), ("fc", 10), ("fc", 2)],
        shifter_net_act_fn=nn.LeakyReLU,
        shifter_net_out_act_fn=nn.Tanh,
        pointwise_conv_config={
            "in_channels": 7334,
            "out_channels": 128,
            "bias": False,
            "batch_norm": False,
            "act_fn": nn.ReLU,
        },
        in_channels_group_size=1, # combine in_channels_group_size channels into 1 by summing
        learn_grid=False,
        grid_l1_reg=0.0,
        grid_net_config={
            "in_channels": 4, # x, y, z, resp
            "layers_config": [("fc", 32), ("fc", 64), ("fc", 16*9)],
            "act_fn": nn.LeakyReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.1,
            "batch_norm": False,
        },
        gauss_blur=True,
        gauss_blur_kernel_size=9,
        gauss_blur_sigma="fixed",
        gauss_blur_sigma_init=1.5,
    ):
        super().__init__()
        
        self.requires_neuron_coords = True
        self.requires_pupil_center = True

        self.H = H
        self.W = W

        self.shift_coords = shift_coords
        self.shifter_net = None
        if shift_coords:
            self.shifter_net = ShifterNet( # pupil center (x, y) -> (delta_x, delta_y)
                in_channels=2,
                layers_config=shifter_net_layers,
                act_fn=shifter_net_act_fn,
                out_act_fn=shifter_net_out_act_fn,
                dropout=0.0,
                batch_norm=False,
            )

        self.in_channels_group_size = in_channels_group_size
        self.pointwise_conv_config = pointwise_conv_config
        if pointwise_conv_config is not None:
            self.pointwise_conv = nn.Sequential(
                nn.Dropout2d(pointwise_conv_config["dropout"]) if pointwise_conv_config.get("dropout", 0) > 0 else nn.Identity(),
                nn.Conv2d(
                    in_channels=int(np.ceil(pointwise_conv_config["in_channels"] / in_channels_group_size)),
                    out_channels=pointwise_conv_config["out_channels"],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=pointwise_conv_config.get("bias", False),
                ),
                nn.BatchNorm2d(pointwise_conv_config["out_channels"]) if pointwise_conv_config.get("batch_norm", False) else nn.Identity(),
                pointwise_conv_config.get("act_fn", nn.ReLU)(),
            )
        self.out_channels = pointwise_conv_config["out_channels"]

        self.learn_grid = learn_grid
        self.grid_l1_reg = grid_l1_reg
        if learn_grid:
            assert grid_net_config is not None, "grid_net_config must be provided if learn_grid is True"
            if gauss_blur:
                print("[WARNING] gauss_blur=True with learn_grid=True is not recommended.")
            self.grid_net_config = grid_net_config
            self.grid_net = build_layers(**grid_net_config)

        ### gauss blur
        assert gauss_blur_sigma in ("fixed", "single", "per_neuron"), \
            "learned_gauss_blur_sigma must be 'fixed', 'single', or 'per_neuron'"
        self.gauss_blur = gauss_blur
        self.gauss_blur_kernel_size = gauss_blur_kernel_size
        self.gauss_blur_sigma_type = gauss_blur_sigma
        self.gauss_blur_sigma_init = gauss_blur_sigma_init
        if gauss_blur_sigma == "single":
            self.gauss_blur_sigma = nn.Parameter(torch.tensor(float(gauss_blur_sigma_init)))
        elif gauss_blur_sigma == "per_neuron":
            self.gauss_blur_sigma = nn.Parameter(torch.ones(pointwise_conv_config["in_channels"]) * gauss_blur_sigma_init)
        else:
            self.gauss_blur_sigma = gauss_blur_sigma_init

    def _apply_gauss_blur(self, x):
        if self.gauss_blur_sigma_type == "fixed":
            return gaussian_blur(x, kernel_size=self.gauss_blur_kernel_size, sigma=self.gauss_blur_sigma)
        else:
            x_cord = torch.arange(self.gauss_blur_kernel_size, device=x.device)
            x_grid = x_cord.repeat(self.gauss_blur_kernel_size).view(self.gauss_blur_kernel_size, self.gauss_blur_kernel_size)
            y_grid = x_grid.T
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

            mean = (self.gauss_blur_kernel_size - 1) / 2.
            variance = self.gauss_blur_sigma**2.

            gaussian_kernel = (1. / (2. * 3.14159265359 * variance)) * torch.exp(
                -torch.sum((xy_grid - mean)**2., dim=-1).unsqueeze(2).expand(-1, -1, variance.numel())
                / (2 * variance)
            )
            gaussian_kernel /= torch.sum(gaussian_kernel, dim=[0, 1], keepdim=True)

            gaussian_kernel = gaussian_kernel.view(-1, 1, self.gauss_blur_kernel_size, self.gauss_blur_kernel_size)
            gaussian_kernel = gaussian_kernel.expand(x.shape[1], -1, -1, -1)

            x = F.conv2d(x, gaussian_kernel, groups=x.shape[1], padding=self.gauss_blur_kernel_size // 2)
            return x

    def _place_responses(self, x, neuron_coords):
        # ### reshape to 3D tensor (sorted by the coordinates)
        # # sort by y and reshape to 3D tensor (B, 3, H, W)
        # neuron_y_sorted_idxs = neuron_coords[:,:,1].argsort(dim=-1)
        # neuron_coords = torch.gather(neuron_coords, 1, neuron_y_sorted_idxs.unsqueeze(-1).expand(-1, -1, 3))
        # x = torch.gather(x, 1, neuron_y_sorted_idxs)
        # x = torch.unflatten(x, dim=1, sizes=(1, 38, 193))
        # neuron_coords = torch.unflatten(neuron_coords, dim=1, sizes=(38, 193))
        
        # # sort by x in each row:
        # neuron_x_sorted_idxs = neuron_coords[:,:,:,0].argsort(-1)
        # x = torch.gather(x, 2, neuron_x_sorted_idxs)
        # # neuron_coords = torch.gather(neuron_coords, -1, neuron_x_sorted_idxs.unsqueeze(-1).expand(-1, -1, -1, 3))
        # # neuron_x_sorted_idxs = neuron_coords[:,:,0].argsort(dim=-1)
        # # neuron_coords = torch.gather(neuron_coords, 2, neuron_x_sorted_idxs.unsqueeze(-1).expand(-1, -1, 3))
        # # x = torch.gather(x, 2, neuron_x_sorted_idxs)
        # # x = torch.unflatten(x, dim=2, sizes=(3, 9, 16))
        # # neuron_coords = torch.unflatten(neuron_coords, dim=2, sizes=(3, 9, 16))

        # ### reshape to 3D tensor (B, 3, H, W)
        # x = torch.unflatten(x, dim=1, sizes=(3, 9, 16))
        # return x


        coords = neuron_coords[..., :2]  # (B, n_neurons, 2)

        ### transform coords from [-1, 1] to [0, W-1] and [0, H-1]
        coords_img_size = (coords + coords.min(1, keepdim=True).values.abs())
        coords_img_size /= coords_img_size.max(1, keepdim=True).values
        coords_img_size = (coords_img_size * torch.tensor([self.W - 1, self.H - 1], device=coords.device)).long()

        ### scatter ones to the positions of the neurons
        B, n_neurons = x.shape[:2]
        pos_x = torch.zeros(B, n_neurons, self.H, self.W, device=x.device)
        pos_x[
            torch.arange(B).unsqueeze(-1).expand(-1, n_neurons),
            torch.arange(n_neurons).unsqueeze(0).expand(B, -1),
            coords_img_size[..., 1].cpu(),
            coords_img_size[..., 0].cpu()
        ] = x  # (B, n_neurons, self.H, self.W)

        return pos_x

    def _combine_in_channels(self, x):
        if self.in_channels_group_size < 2:
            return x

        B, n_neurons, H, W = x.shape

        ### pad x to have in_channels_group_size as a divisor
        pad_to = np.ceil(n_neurons / self.in_channels_group_size) * self.in_channels_group_size
        x = F.pad(x, (0, 0, 0, 0, 0, int(pad_to - n_neurons)), "constant", 0)

        ### combine in_channels_group_size channels into 1 by summing
        x = x.view(B, -1, self.in_channels_group_size, H, W)
        x = x.sum(dim=2) # (B, n_neurons / in_channels_group_size, H, W)

        return x

    def set_additional_loss(self, inp, out):
        if hasattr(self, "decoding_loss_fn"):
            self._last_loss = self.decoding_loss_fn(
                self.decoder(out["encoded"].view(inp["resp"].size(0), -1)),
                inp["resp"]
            )
        elif hasattr(self, "grid_l1_reg") and hasattr(self, "pos_x") and self.grid_l1_reg > 0:
            self._last_loss = self.grid_l1_reg * self.pos_x.abs().sum(dim=(-1,-2)).mean()

    def forward(self, x, neuron_coords, pupil_center):
        B, n_neurons = x.shape

        ### prepare neuron_coords
        n_coords = neuron_coords.size(-1)
        if neuron_coords.ndim == 2:
            neuron_coords = neuron_coords.unsqueeze(0).repeat(B, 1, 1) # (B, n_neurons, n_coords)
        if self.shift_coords:
            ### shift neuron_coords by pupil_center
            delta = self.shifter_net(pupil_center) # (B, 2)
            neuron_coords[:, torch.arange(n_neurons), :2] += delta.unsqueeze(1)

        ### construct positioned responses tensor (B, n_neurons, H, W) based on neuron coords
        if self.learn_grid:
            grid_net_inp = torch.cat([
                neuron_coords,
                torch.log10(x.clamp_min(1e-3)).unsqueeze(-1)
            ], dim=-1) # (B, n_neurons, n_coords + 1)
            grid_net_inp = grid_net_inp.view(B * n_neurons, n_coords + 1)
            pos_x = self.grid_net(grid_net_inp) # (B * n_neurons, H * W)
            pos_x = pos_x.view(B, n_neurons, -1).view(B, n_neurons, self.H, self.W)
            self.pos_x = pos_x if self.grid_l1_reg > 0 else None
        else:
            pos_x = self._place_responses(x, neuron_coords) # (B, n_neurons, H, W)

        ### apply gaussian filter
        if self.gauss_blur:
            pos_x = self._apply_gauss_blur(pos_x)

        ### combine in_channels_group_size channels into 1 by summing
        pos_x = self._combine_in_channels(pos_x)

        ### apply pointwise conv
        if self.pointwise_conv_config is not None:
            pos_x = self.pointwise_conv(pos_x)

        return pos_x


class Conv1dReadIn(ReadIn):
    def __init__(
        self,
        in_shape,
        layers_config=[
            ("conv1d", 64, 7, 3, 3),
            ("conv1d", 32, 7, 3, 3),
            ("conv1d", 16, 5, 2, 2),
            ("conv1d", 8, 4, 2, 1),
            ("flatten", 1, -1, 1632),
            ("fc", 288),
            ("unflatten", 1, (2, 9, 16)),
        ],
        act_fn=nn.LeakyReLU,
        out_act_fn=nn.Identity,
        dropout=0.1,
        batch_norm=False,
        out_channels=None,
    ):
        super().__init__()
        self.requires_neuron_coords = False
        self.requires_pupil_center = False

        self.layers = build_layers(
            in_channels=in_shape,
            layers_config=layers_config,
            act_fn=act_fn,
            out_act_fn=out_act_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = layers_config[-1][-1]

    def forward(self, x, neuron_coords=None, pupil_center=None):
        return self.layers(x.unsqueeze(1))


class Attention(nn.Module):
    def __init__(
        self,
        model_dim,
        heads=2,
        dim_head=16, # dim of q, k, v
        dropout=0.,
        causal=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(model_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (B, N, inner_dim * 3) -> (B, N, inner_dim) x 3
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if self.causal:
            # apply causal mask
            mask = torch.ones(size=sim.shape[-2:], device=sim.device).triu_(1).bool()
            sim.masked_fill_(mask, float("-inf"))

        attn = sim.softmax(dim=-1)  # (B, H, Q, K)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads) # merge heads
        return self.to_out(out)


class AttentionReadIn(ReadIn):
    """
    - Takes in:
        - neuronal responses (B, N_neurons),
        - neuronal coordinates (N_neurons, 3),
        - pupil center (B, 2).
    - Merges the responses with the (shifted) coordinates into a (B, N_neurons, 4) tensor,
    tokenizes it, getting a (B, ceil(N_neurons / token_neurons), token_neurons * 4) tensor,
    and applies a sequence of [self-attention, down projection] blocks followed by a linear/conv layer.
    - The pupil center is fed through a separate FC layer to get $Delta_x$ and $Delta_y$ that
    are added to the coordinates to shift the receptive field.
    """
    def __init__(
        self,
        in_shape,
        shift_coords=True,
        shifter_net_layers=[("fc", 10), ("fc", 10), ("fc", 2)],
        shifter_net_act_fn=nn.LeakyReLU,
        shifter_net_out_act_fn=nn.Tanh,
        attn_config={
            "layers": 2,
            "token_neurons": 100,
            "dropout": 0.0,
            "attn_num_heads": 4,
        },
        attn_interleave_config={
            "layers": [("fc", 512), ("act_fn", nn.ReLU), ("dropout", 0.0), ("fc", 400)],
            "after_last": False,
        },
        neuron_embed_dim=None,
        conv_out_config=None,
    ):
        super().__init__()
        self.requires_neuron_coords = True
        self.requires_pupil_center = True
        self.in_shape = in_shape

        ### shifter net
        self.shift_coords = shift_coords
        self.shifter_net = None
        if shift_coords:
            self.shifter_net = ShifterNet( # pupil center (x, y) -> (delta_x, delta_y)
                in_channels=2,
                layers_config=shifter_net_layers,
                act_fn=shifter_net_act_fn,
                out_act_fn=shifter_net_out_act_fn,
                dropout=0.0,
                batch_norm=False,
            )

        ### neuronal embeddings
        self.neuron_embed_dim = neuron_embed_dim
        if self.neuron_embed_dim:
            self.neuron_embed = nn.Embedding(
                num_embeddings=in_shape,
                embedding_dim=neuron_embed_dim,
            )

        ### attention
        self.attn_config = attn_config
        self.attn_layers = []
        self.model_dim = attn_config["token_neurons"] * 4
        if self.neuron_embed_dim:
            self.model_dim += attn_config["token_neurons"] * neuron_embed_dim
        for attn_l_idx in range(attn_config["layers"]):
            self.attn_layers.append(Attention(
                model_dim=self.model_dim,
                heads=attn_config["attn_num_heads"],
                dim_head=attn_config["dim_head"],
                dropout=attn_config["dropout"],
                causal=False,
            ))
            curr_dim = self.model_dim
            for layer_config in attn_interleave_config["layers"]:
                if attn_l_idx == attn_config["layers"] - 1 and not attn_interleave_config["after_last"]:
                    break
                if layer_config[0] == "fc":
                    self.attn_layers.append(nn.Linear(curr_dim, layer_config[1]))
                    curr_dim = layer_config[1]
                elif layer_config[0] == "act_fn":
                    self.attn_layers.append(layer_config[1]())
                elif layer_config[0] == "dropout":
                    self.attn_layers.append(nn.Dropout(layer_config[1]))
                else:
                    raise ValueError(f"layer_type {layer_config[0]} not recognized")
        self.attn = nn.Sequential(*self.attn_layers)
        self.n_tokens = int(np.ceil(in_shape / attn_config["token_neurons"]))

        ### set out channels (experimentally)
        self.out_dim = (18, 32)
        test_inp = torch.randn(1, in_shape, self.model_dim)
        test_inp = self._tokenize(test_inp)
        test_inp = self.attn(test_inp)
        test_inp = self._reshape_out(test_inp)
        self.out_channels = test_inp.size(1)

        self.conv_out_config = conv_out_config
        if conv_out_config is not None:
            self.conv_out = []
            self.conv_out.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=conv_out_config["out_channels"],
                kernel_size=conv_out_config["kernel_size"],
                stride=conv_out_config["stride"],
                padding=conv_out_config["padding"],
                bias=conv_out_config.get("bias", False),
            ))
            if conv_out_config.get("batch_norm", False):
                self.conv_out.append(nn.BatchNorm2d(conv_out_config["out_channels"]))
            self.conv_out.append(conv_out_config.get("act_fn", nn.ReLU)())
            self.conv_out = nn.Sequential(*self.conv_out)
            self.out_channels = conv_out_config["out_channels"]


    def _reshape_out(self, x):
        B = x.size(0)
        x = x.view(B, -1)  # (B, n_neurons * self.model_dim)
        target_size = (x.shape[1:].numel() // np.prod(self.out_dim) + 1) * np.prod(self.out_dim)
        x = F.pad(x, (0, target_size - x.shape[1]), "constant", 0)
        x = x.view(B, -1, *self.out_dim)
        return x

    def _tokenize(self, x):
        B = x.shape[0]
        x = x.view(B, -1)  # (B, n_neurons * self.model_dim)

        ### pad last token with zeros
        curr_dim, target_dim = x.shape[1], self.n_tokens * self.model_dim
        x = F.pad(x, (0, target_dim - curr_dim), "constant", 0)
        x = x.view(B, self.n_tokens, self.model_dim)

        return x

    def forward(self, x, neuron_coords, pupil_center):
        B, n_neurons = x.shape

        neuron_coords = neuron_coords.unsqueeze(0).repeat(B, 1, 1) # (B, n_neurons, 3)

        if self.shift_coords:
            ### shift neuron_coords by pupil_center
            delta = self.shifter_net(pupil_center) # (B, 2)
            neuron_coords[:, torch.arange(n_neurons), :2] += delta.unsqueeze(1)

        ### merge responses with (shifted) coordinates
        x = torch.cat([x.unsqueeze(-1), neuron_coords], dim=-1)  # (B, n_neurons, 4)
        if self.neuron_embed_dim:
            neuron_embed = self.neuron_embed(torch.arange(n_neurons, device=x.device))
            x = torch.cat([x, neuron_embed.unsqueeze(0).repeat(B, 1, 1)], dim=-1)  # (B, n_neurons, 4 + neuron_embed_dim)

        ### tokenize
        x = self._tokenize(x)  # (B, self.n_tokens, self.model_dim)

        ### self-attention
        x = self.attn(x)  # (B, self.n_tokens, self.model_dim)

        ### reshape for output
        x = self._reshape_out(x)  # (B, -1, 18, 32)

        ### apply conv layer
        if self.conv_out_config is not None:
            x = self.conv_out(x)

        return x

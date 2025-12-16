import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from csng.models.gan import GAN
from csng.utils.mix import build_layers
from csng.utils.data import crop


class MultiReadIn(nn.Module):
    """Decoder that consists of a single core module and multiple ReadIn blocks for different data keys (data key ~ dataset name).
    
    :param readins_config: list of dictionaries, each containing the configuration of a ReadIn block.
    :param core_cls: class of the core
    :param core_config: configuration of the core
    :param crop_stim_fn (optional): function to crop the stimulus after the core block
    """

    def __init__(self, readins_config, core_cls, core_config, crop_stim_fn=None):
        super().__init__()

        ### init all readin layers
        self.readins_config = readins_config # 1 ReadIn <=> 1 FC layer
        self.readins = nn.ModuleDict()
        for readin_config in readins_config:
            self.readins[readin_config["data_key"]], in_channels = self._create_readin(readin_config)

        ### core decoder
        self.core_config = deepcopy(core_config)
        self.core = core_cls(**self.core_config)

        ### crop stim
        self.crop_stim_fn = crop_stim_fn

    def _create_readin(self, readin_config):
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
        
        return nn.ModuleList(readin), in_channels

    def load_from_ckpt(self, ckpt, load_best, load_only_core, strict=True):
        assert not strict or not load_only_core, "strict=True cannot be used with load_only_core=True"

        ### prepare what state_dict to load
        if load_best:
            to_load = ckpt["best"]["decoder"]
        else:
            to_load = ckpt["decoder"]

        if load_only_core:
            to_load = {k:v for k,v in to_load.items() if "readin" not in k.lower()}

        ### load state_dict
        self._load_state_dict(to_load, strict=strict)

    def _load_state_dict(self, state_dict, strict=True):
        if self.core.__class__ == GAN:
            print(f"[WARNING] Loading GAN state_dict without the optimizer states.")
            self.core.load_state_dict(state_dict={".".join(k.split(".")[1:]):v for k,v in state_dict.items() if "readin" not in k.lower()}, load_optimizers=False)
            self.readins.load_state_dict({".".join(k.split(".")[1:]):v for k,v in state_dict.items() if "readin" in k.lower()}, strict=strict)
        else:
            raise ValueError(f"core_cls {self.core.__class__} not recognized")

    def add_readin(self, data_key, readin_config):
        if data_key in self.readins:
            print(f"[WARNING] readin layer for {data_key} already exists, overwriting it.")
        self.readins[data_key], _ = self._create_readin(readin_config)
        self.readins_config.append(readin_config)

    def set_additional_loss(self, inp, out):
        for readin_data_key, readin in self.readins.items():
            if "data_key" in inp and inp["data_key"] != readin_data_key:
                continue
            encoded = inp["resp"]
            for l in readin:
                if isinstance(l, ReadIn):
                    encoded = l(encoded, neuron_coords=inp["neuron_coords"] if "neuron_coords" in inp else None,
                                pupil_center=inp["pupil_center"] if "pupil_center" in inp else None)
                else:
                    encoded = l(encoded)
            out["encoded"] = encoded
            for l in readin:
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

    def forward(self, x, data_key=None, neuron_coords=None, pupil_center=None, additional_core_inp=None, additional_core_channels=None):
        """
        :param x: neuronal responses (B, N_neurons)
        :param data_key: data key to select the readin
        :param neuron_coords: neuronal coordinates (B, N_neurons, 2 or 3) or None if not used by the readins
        :param pupil_center: pupil center (B, 2) or None if not used by the readins
        :param additional_core_inp: additional input when calling the core block
        :param additional_core_channels: additional channels to concatenate to the core input
        :return: stimulus reconstruction (B, C, H, W)
        """
        assert data_key is not None or len(self.readins) == 0, \
            "data_key must be provided if there are multiple readins"

        ### run readin layer(s)
        if data_key is not None:
            for l in self.readins[data_key]:
                if isinstance(l, ReadIn):
                    x = l(x, neuron_coords=neuron_coords[data_key] if isinstance(neuron_coords, dict) else neuron_coords, pupil_center=pupil_center)
                else:
                    x = l(x)

        ### run core decoder
        if additional_core_channels is not None:
            x = torch.cat([x, additional_core_channels], dim=1)
        x = self.core(x) if additional_core_inp is None else self.core(x, **additional_core_inp)
        if self.crop_stim_fn:
            x = self.crop_stim_fn(x)

        return x


class ReadIn(nn.Module):
    """Base class for ReadIn blocks."""

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


class ShifterNet(nn.Module):
    """Fully connected network that takes in pupil center coordinates and outputs the shift in the receptive field.

    :param in_channels: input shape
    :param layers_config: list of tuples, each containing the configuration of a layer (see build_layers)
    :param act_fn: activation function
    :param out_act_fn: output activation function
    :param dropout: dropout probability
    :param batch_norm: whether to use batch normalization
    """

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


class MEIReadIn(ReadIn):
    """Most Exciting Input (MEI) ReadIn block.

    Contextually modulates MEIs based on the responses (and neuron coordinates),
    and then applies a pointwise convolutional layer to the modulated MEIs.

    :param meis_path: path to the MEIs
    :param n_neurons: number of neurons
    :param mei_target_shape: target shape of the MEIs (H, W)
    :param mei_resize_method: method to resize the MEIs (crop or resize)
    :param meis_trainable: whether the MEIs are trainable, i.e. optimized along with other parameters
    :param pointwise_conv_config: configuration of the pointwise convolutional layer (set to None if not used)
    :param ctx_net_config: configuration of the context network
    :param apply_resp_transform: whether to apply the clamp-log10 transformation to the responses before the first layer
    :param shift_coords: whether to shift the MEIs based on the pupil center
    :param shifter_net_layers: list of tuples, each containing the configuration of a layer of the shifter network
    :param shifter_net_act_fn: activation function of the shifter network
    :param shifter_net_out_act_fn: output activation function of the shifter network
    :param out_channels: number of output channels (if None, it is set automatically)
    :param neuron_idxs: selection of neurons to use (if None, all neurons are used)
    :param device: device
    """

    def __init__(
        self,
        meis_path,
        n_neurons,
        mei_target_shape,
        mei_resize_method="resize",
        meis_trainable=False,
        use_neuron_coords=False,
        pointwise_conv_config={
            "out_channels": 256,
            "bias": False,
            "batch_norm": False,
            "act_fn": nn.ReLU,
        },
        ctx_net_config={
            "in_channels": 3, # resp, x, y
            "layers_config": [("fc", 32), ("fc", 128), ("fc", 22*36)],
            "act_fn": nn.LeakyReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.1,
            "batch_norm": False,
        },
        apply_resp_transform=False,
        shift_coords=False,
        shifter_net_layers=[("fc", 10), ("fc", 10), ("fc", 2)],
        shifter_net_act_fn=nn.LeakyReLU,
        shifter_net_out_act_fn=nn.Tanh,
        neuron_emb_dim=None,  # dim of learned neuron embeddings (None if not used)
        out_channels=None, # set manually
        neuron_idxs=None, # selection of neurons to use
        l2_reg_mul=0.0,
        l1_reg_mul=0.0,
        device="cpu",
    ):
        super().__init__()
        
        self.requires_neuron_coords = True
        self.requires_pupil_center = True
        self.register_buffer("neuron_idxs", torch.tensor(neuron_idxs) if neuron_idxs is not None else None)
        self.n_neurons = n_neurons if neuron_idxs is None else len(neuron_idxs)
        self.use_neuron_coords = use_neuron_coords
        self.l2_reg_mul = l2_reg_mul
        self.l1_reg_mul = l1_reg_mul
        self.device = device
        assert not self.use_neuron_coords or ctx_net_config["in_channels"] > 1

        ### setup MEIs
        self.meis_path = meis_path
        self.meis = torch.load(meis_path)["meis"].to(device)
        if self.neuron_idxs is not None:
            self.meis = self.meis[self.neuron_idxs]
        assert self.meis.shape[0] == self.n_neurons, "number of neurons in MEIs does not match n_neurons"
        if mei_resize_method == "crop":
            self.meis = crop(self.meis, mei_target_shape)
        elif mei_resize_method == "resize":
            self.meis = torchvision.transforms.Resize(mei_target_shape)(self.meis)
        else:
            raise ValueError(f"mei_resize_method {mei_resize_method} not recognized")
        self.meis = self.meis.permute(1,0,2,3) # (1, n_neurons, H, W)
        if meis_trainable:
            self.meis = nn.Parameter(self.meis)

        ### shifter network
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

        ### pointwise convolutional layer
        self.pointwise_conv_config = pointwise_conv_config
        self.pointwise_conv = nn.Identity()
        if pointwise_conv_config is not None:
            self.pointwise_conv = nn.Sequential(
                nn.Dropout2d(pointwise_conv_config["dropout"]) if pointwise_conv_config.get("dropout", 0) > 0 else nn.Identity(),
                nn.Conv2d(
                    in_channels=self.n_neurons,
                    out_channels=pointwise_conv_config["out_channels"],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=pointwise_conv_config.get("bias", False),
                ),
                nn.BatchNorm2d(pointwise_conv_config["out_channels"]) if pointwise_conv_config.get("batch_norm", False) else nn.Identity(),
                pointwise_conv_config.get("act_fn", nn.ReLU)(),
            )
        self.out_channels = pointwise_conv_config["out_channels"] if out_channels is None else out_channels

        ### context network
        self.resp_transform = self._resp_transform if apply_resp_transform else nn.Identity()
        self.ctx_net_config = ctx_net_config
        self.ctx_net = build_layers(**ctx_net_config)

        ### learned neuron embeddings
        self.neuron_emb_dim = neuron_emb_dim
        if neuron_emb_dim:
            self.neuron_embed = nn.Embedding(
                num_embeddings=self.n_neurons,
                embedding_dim=neuron_emb_dim,
            )

    @staticmethod
    def _resp_transform(x):
        return torch.log10(x.clamp_min(1e-3))

    def set_additional_loss(self, inp, out):
        self._last_loss = 0.
        if self.l2_reg_mul > 0:
            self._last_loss += self.l2_reg_mul * sum(p.pow(2).sum() for p in self.parameters())
        if self.l1_reg_mul > 0:
            self._last_loss += self.l1_reg_mul * sum(p.abs().sum() for p in self.parameters())

    def forward(self, x, neuron_coords, pupil_center):
        """
        :param x: neuronal responses (B, N_neurons)
        :param neuron_coords: neuronal coordinates (B, N_neurons, 2 or 3) or None if not used
        :param pupil_center: pupil center (B, 2) or None if shift_coords=False

        :return: channel reduced contextualized MEIs (B, N_reduced, H, W)
        """
        ### filter neurons
        if self.neuron_idxs is not None:
            x = x[..., self.neuron_idxs]
            neuron_coords = neuron_coords[..., self.neuron_idxs, :] if neuron_coords is not None else None

        B, n_neurons = x.shape

        ### prepare neuron coordinates
        if self.use_neuron_coords and self.ctx_net_config["in_channels"] != 1 and neuron_coords.ndim == 2:
            neuron_coords = neuron_coords.unsqueeze(0).repeat(B, 1, 1)
        if self.shift_coords:
            ### shift neuron_coords by pupil_center
            delta = self.shifter_net(pupil_center)
            neuron_coords[:, torch.arange(n_neurons), :2] += delta.unsqueeze(1)

        ### prepare MEIs
        out = self.meis.expand(B, -1, -1, -1) # (B, n_neurons, H, W)
        if out.device != x.device: # fix for data parallelism
            out = out.to(x.device)

        ### contextually modulate MEIs based on the responses (and other inputs)      
        ctx_inp = [self.resp_transform(x).unsqueeze(-1)]
        if self.use_neuron_coords:
            ctx_inp.append(neuron_coords[..., :2])
        if self.neuron_emb_dim:
            neuron_embeds = self.neuron_embed(torch.arange(n_neurons, device=x.device))
            neuron_embeds = neuron_embeds.unsqueeze(0).repeat(B, 1, 1)
            ctx_inp.append(neuron_embeds)
        ctx_inp = torch.cat(ctx_inp, dim=-1).view(B * n_neurons, -1) # (B * n_neurons, D)
        ctx_out = self.ctx_net(ctx_inp) # (B * n_neurons, H * W)
        out = out * ctx_out.view(B, n_neurons, *out.shape[-2:])

        ### apply pointwise conv
        out = self.pointwise_conv(out)

        return out

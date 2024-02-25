import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import grad

from csng.utils import normalize, standardize
from csng.utils import build_layers
# from mypkg.timing import timeit


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        act_fn1=nn.ReLU(),
        act_fn2=nn.ReLU(),
        batch_norm1=False,
        batch_norm2=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if in_channels != out_channels: # use 1x1 conv to match dimensions
            self.conv_skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        self.act_fn1 = act_fn1
        self.act_fn2 = act_fn2
        self.batch_norm1 = batch_norm1
        self.batch_norm2 = batch_norm2
        if self.batch_norm1:
            self.bn1 = nn.BatchNorm2d(out_channels)
        if self.batch_norm2:
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_hat = self.conv1(x)
        if self.batch_norm1:
            x_hat = self.bn1(x_hat)
        x_hat = self.act_fn1(x_hat)
        
        x_hat = self.conv2(x_hat)
        if self.batch_norm2:
            x_hat = self.bn2(x_hat)
        x_hat = self.act_fn2(x_hat)
        
        if self.in_channels != self.out_channels:
            x = self.conv_skip(x)
        
        return x_hat + x


class L2O_Decoder(nn.Module):
    def __init__(
        self,
        encoder,
        resp_shape=(10000,),
        stim_shape=(1, 50, 50),
        in_shape=(3, 50, 50),
        resp_layers_cfg={
            "layers": [("fc", 384), ("unflatten", 1, (6, 8, 8)), ("deconv", 128, 7, 2, 0), ("deconv", 64, 5, 2, 0)],
            "batch_norm": True,
            "dropout": 0.0,
            "act_fn": nn.ReLU(),
            "out_act_fn": nn.Identity(),
        },
        reconstruction_init_method="resp_layers",
        act_fn=nn.ReLU(),
        stim_loss_fn=nn.MSELoss(),
        resp_loss_fn=nn.MSELoss(),
        opter_cls=torch.optim.Adam,
        opter_kwargs={},
        unroll=5,
        preproc_grad=True,
        device="cpu",
    ):
        assert reconstruction_init_method in ["resp_layers", "random", "zero"], \
            "reconstruction_init_method must be one of: 'resp_layers', 'random' or 'zero'."

        super().__init__()
        self.encoder = encoder.eval()
        self.encoder.training = False
        self.encoder.requires_grad_(False)
        self.resp_shape = resp_shape
        self.stim_shape = stim_shape
        self.in_shape = in_shape
        self.resp_layers_cfg = resp_layers_cfg
        self.reconstruction_init_method = reconstruction_init_method
        self.act_fn = act_fn
        self.stim_loss_fn = stim_loss_fn
        self.resp_loss_fn = resp_loss_fn
        self.unroll = unroll
        self.preproc_grad = preproc_grad
        self.device = device

        ### build layers
        self.hidden_size = 128
        self.bottleneck_in_shape = (16, 2, 6)
        self.bottleneck_out_shape = (16, 2, 6)

        if self.reconstruction_init_method == "resp_layers":
            self._build_resp_layers()
        else:
            self.resp_layers = None
        self._build_conv_reconstruction_layers(
            bottleneck_in_channels=self.bottleneck_in_shape[0],
            bottleneck_out_channels=self.bottleneck_out_shape[0],
        )
        self._build_rec_bottleneck(
            in_size=np.prod(self.bottleneck_in_shape),
            out_size=np.prod(self.bottleneck_out_shape),
            hidden_size=self.hidden_size,
        )

        self.opter = opter_cls(self.parameters(), **opter_kwargs)

    def _build_resp_layers(self):
        layers = []
        in_channels = self.resp_shape[0]

        ### build layers from config
        for l_i, layer_config in enumerate(self.resp_layers_cfg["layers"]):
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

            if l_i < len(self.resp_layers_cfg["layers"]) - 1 and layer_type in ["fc", "conv", "deconv"]:
                ### add batch norm, activation, dropout
                if self.resp_layers_cfg["batch_norm"]:
                    if layer_type == "fc":
                        layers.append(nn.BatchNorm1d(out_channels))
                    else:
                        layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.resp_layers_cfg["act_fn"])
                if self.resp_layers_cfg["dropout"] > 0.0:
                    layers.append(nn.Dropout(self.resp_layers_cfg["dropout"]))
            elif l_i == len(self.resp_layers_cfg["layers"]) - 1:
                ### add output activation
                layers.append(self.resp_layers_cfg["out_act_fn"])

            in_channels = out_channels

        self.resp_layers = nn.Sequential(*layers)

    def _build_rec_bottleneck(self, in_size, out_size, hidden_size):
        ### recurrent bottleneck
        self.rec1 = nn.LSTMCell(input_size=in_size, hidden_size=hidden_size)
        self.rec2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.rec_out = nn.Linear(in_features=hidden_size, out_features=out_size)

    def _build_conv_reconstruction_layers(self, bottleneck_in_channels=32, bottleneck_out_channels=32):
        ### conv blocks down
        self.down_act_fn = self.act_fn
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.block_in = nn.Conv2d(
            in_channels=self.in_shape[0],
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )

        self.block_down1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_down2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_down3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_down4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_to_bottleneck = nn.Conv2d(
            in_channels=64,
            out_channels=bottleneck_in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        ### conv blocks up (after recurrent bottleneck)
        self.up_act_fn = self.act_fn
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        self.block_up1 = nn.Conv2d(
            in_channels=bottleneck_out_channels + self.block_down4.out_channels, # residual connection
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.block_up2 = nn.Conv2d(
            in_channels=64 + self.block_down3.out_channels, # residual connection
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_up3 = nn.Conv2d(
            in_channels=64 + self.block_down2.out_channels, # residual connection
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_up4 = nn.Conv2d(
            in_channels=64 + self.block_down1.out_channels, # residual connection
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_up5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        self.block_out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding="same",
        )

    def _init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, self.rec1.hidden_size, device=self.device),
            torch.zeros(batch_size, self.rec2.hidden_size, device=self.device),
        )
    
    def _init_cell(self, batch_size):
        return (
            torch.zeros(batch_size, self.rec1.hidden_size, device=self.device),
            torch.zeros(batch_size, self.rec2.hidden_size, device=self.device),
        )

    # @timeit
    def _get_resp_err_grad(self, x_hat, resp_target, data_key):
        resp_x_hat = self.encoder(normalize(x_hat, mean=0.441, std=0.254), data_key=data_key) # TODO: quick hack
        resp_loss = self.resp_loss_fn(resp_x_hat, resp_target)
        x_hat_grad = grad(resp_loss, x_hat, create_graph=False)[0]
        return x_hat_grad

    # @timeit
    def _preproc_grad(self, grad, preproc_factor=10.0):
        B, C, H, W = grad.shape
        preproc_threshold = np.exp(-preproc_factor)
        grad = grad.data # (B, C, H, W)

        grad_preproc = torch.zeros(B, C * 2, H, W, device=self.device)
        keep_grad_mask = (torch.abs(grad) >= preproc_threshold) # (B, C, H, W)
        
        ### if gradient is large enough, set to log(abs(grad)) / preproc_factor and sign(grad)
        grad_preproc[:, 0][keep_grad_mask.squeeze(1)] = (
            torch.log(torch.abs(grad[keep_grad_mask]) + 1e-8) / preproc_factor
        ).squeeze()
        grad_preproc[:, 1][keep_grad_mask.squeeze(1)] = (
            torch.sign(grad[keep_grad_mask]).squeeze()
        )

        ### if gradient is too small, set to -1 and exp(preproc_factor) * grad
        grad_preproc[:, 0][~keep_grad_mask.squeeze(1)] = -1
        grad_preproc[:, 1][~keep_grad_mask.squeeze(1)] = (
            float(np.exp(preproc_factor)) * grad[~keep_grad_mask]
        ).squeeze()
        
        return grad_preproc

    def _pad_concat(self, x1, x2):
        """ Pad x1 to match x2's shape and concatenate along the channel dimension. """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)

    # @timeit
    def _init_reconstruction(self, resp):
        if self.reconstruction_init_method == "resp_layers":
            x_hat = self.resp_layers(resp)
        elif self.reconstruction_init_method == "random":
            x_hat = torch.rand(resp.shape[0], *self.stim_shape, device=self.device)
        elif self.reconstruction_init_method == "zero":
            x_hat = torch.zeros(resp.shape[0], *self.stim_shape, device=self.device)
        return x_hat

    # @timeit
    def forward_single(self, x_hat, x_hat_grad, x_hat_t0, hidden, cell):
        ### construct input
        x = torch.cat([x_hat, x_hat_grad, x_hat_t0], dim=1) # (B, C*3, H, W)

        ### conv blocks down
        x = self.down_act_fn(self.block_in(x)) # (50, 50)
        
        x = self.maxpool(x) # (25, 25)
        conv1 = self.down_act_fn(self.block_down1(x)) # (25, 25)

        x = self.maxpool(conv1) # (12, 12)
        conv2 = self.down_act_fn(self.block_down2(x)) # (12, 12)

        x = self.maxpool(conv2) # (6, 6)
        conv3 = self.down_act_fn(self.block_down3(x)) # (6, 6)

        # x = self.maxpool(conv3) # (6, 6)
        x = conv3
        conv4 = self.down_act_fn(self.block_down4(x)) # (6, 6)
        
        x = self.down_act_fn(self.block_to_bottleneck(conv4)) # self.bottleneck_in_shape

        ### recurrent bottleneck
        (h1, h2), (c1, c2) = hidden, cell
        x = x.view(x.shape[0], np.prod(self.bottleneck_in_shape)) # (B, C*H*W)
        h1, c1 = self.rec1(x, (h1, c1))
        h2, c2 = self.rec2(h1, (h2, c2))
        x = self.rec_out(h2)
        x = x.view(x.shape[0], *self.bottleneck_out_shape)

        ### conv blocks up
        x = self._pad_concat(x, conv4) # (6, 6)
        x = self.up_act_fn(self.block_up1(x)) # (6, 6)

        # x = self.upsample(x) # (12, 12)
        x = self._pad_concat(x, conv3) # (6, 6)
        x = self.up_act_fn(self.block_up2(x)) # (6, 6)

        x = self.upsample(x) # (12, 12)
        x = self._pad_concat(x, conv2) # (12, 12)
        x = self.up_act_fn(self.block_up3(x)) # (12, 12)

        x = self.upsample(x) # (24, 24)
        x = self._pad_concat(x, conv1) # (25, 25)
        x = self.up_act_fn(self.block_up4(x)) # (25, 25)

        x = self.upsample(x) # (50, 50)
        x = self.up_act_fn(self.block_up5(x)) # (50, 50)

        x = self.block_out(x) # (50, 50)

        return x, (h1, h2), (c1, c2)

    def forward(self, x, resp, stim, data_key, n_steps=1, train=False, x_hat_history_iters=None, neuron_coords=None, pupil_center=None):
        """ x received from a ReadIn or x == resp """
        assert not train or (train and stim is not None), \
            "You must provide stim during training."

        if train:
            self.train()
            unroll = self.unroll
        else:
            self.eval()
            unroll = 1

        ### init decoded imgs and hidden states
        B = resp.shape[0]
        x_hat = self._init_reconstruction(resp=x) # (B, C, H, W)
        if self.reconstruction_init_method in ("random", "zero"):
            x_hat.requires_grad = True
        else:
            x_hat.retain_grad()
        x_hat_t0 = x_hat.detach().clone()
        hidden = self._init_hidden(batch_size=B) # (B, hidden_size)
        cell = self._init_cell(batch_size=B) # (B, hidden_size)

        ### optimize decoded img
        unroll_loss = 0.
        history = {"x_hat_history": [], "loss": []}
        for step_i in range(1, n_steps + 1):
            ### get the gradient of the response error with respect to the current decoded image
            x_hat.requires_grad_(True)
            x_hat_grad = self._get_resp_err_grad(x_hat=x_hat, resp_target=resp, data_key=data_key).detach()
            if self.preproc_grad:
                x_hat_grad = self._preproc_grad(x_hat_grad)

            ### update the decoded image
            x_hat_delta, hidden, cell = self.forward_single(
                x_hat=x_hat,
                x_hat_grad=x_hat_grad,
                x_hat_t0=x_hat_t0,
                hidden=hidden,
                cell=cell,
            )
            x_hat = F.sigmoid(x_hat + x_hat_delta)

            ### compute current reconstruction loss
            if stim is not None:
                curr_loss = self.stim_loss_fn(x_hat, stim, data_key=data_key)
                unroll_loss += curr_loss
            else:
                curr_loss = None

            ### finish unroll
            if step_i % unroll == 0:
                if train:
                    ### update decoder parameters
                    self.opter.zero_grad()
                    unroll_loss.backward()
                    self.opter.step()
                unroll_loss = 0.

                hidden = tuple(h.detach() for h in hidden)
                cell = tuple(c.detach() for c in cell)
                x_hat = x_hat.detach()
                x_hat.requires_grad = True

            ### log
            history["loss"].append(curr_loss.detach().cpu() if curr_loss is not None else curr_loss)
            if x_hat_history_iters is not None and step_i in x_hat_history_iters:
                history["x_hat_history"].append(x_hat.detach().cpu())

        return x_hat, history


class L2O_Shallow_Decoder(nn.Module):
    def __init__(
        self,
        encoder,
        resp_shape=(10000,),
        stim_shape=(1, 50, 50),
        in_shape=(3, 50, 50),
        resp_layers_cfg={
            "layers": [("fc", 384), ("unflatten", 1, (6, 8, 8)), ("deconv", 128, 7, 2, 0), ("deconv", 64, 5, 2, 0)],
            "batch_norm": True,
            "dropout": 0.0,
            "act_fn": nn.ReLU(),
            "out_act_fn": nn.Identity(),
        },
        reconstruction_init_method="resp_layers",
        reconstruction_layers_cfg={
            "layers": [("conv", 64, 5, 1, 2), ("conv", 64, 5, 1, 2), ("conv", 64, 5, 1, 2)],
            "batch_norm": True,
            "dropout": 0.0,
            "act_fn": nn.ReLU(),
            "out_act_fn": nn.Identity(),
        },
        act_fn=nn.ReLU(),
        stim_loss_fn=nn.MSELoss(),
        resp_loss_fn=nn.MSELoss(),
        opter_cls=torch.optim.Adam,
        opter_kwargs={},
        unroll=5,
        preproc_grad=True,
        device="cpu",
    ):
        assert reconstruction_init_method in ["resp_layers", "random", "zero"], \
            "reconstruction_init_method must be one of: 'resp_layers', 'random' or 'zero'."

        super().__init__()
        self.encoder = encoder.eval()
        self.encoder.training = False
        self.encoder.requires_grad_(False)
        self.resp_shape = resp_shape
        self.stim_shape = stim_shape
        self.in_shape = in_shape
        self.resp_layers_cfg = resp_layers_cfg
        self.reconstruction_init_method = reconstruction_init_method
        self.act_fn = act_fn
        self.stim_loss_fn = stim_loss_fn
        self.resp_loss_fn = resp_loss_fn
        self.unroll = unroll
        self.preproc_grad = preproc_grad
        self.device = device

        ### build layers
        if self.reconstruction_init_method == "resp_layers":
            self._build_resp_layers()
        else:
            self.resp_layers = None

        self.reconstruction_layers_cfg = reconstruction_layers_cfg
        self.recon_layers = build_layers(
            in_channels=self.in_shape[0],
            layers_config=self.reconstruction_layers_cfg["layers"],
            act_fn=self.reconstruction_layers_cfg["act_fn"],
            out_act_fn=self.reconstruction_layers_cfg["out_act_fn"],
            batch_norm=self.reconstruction_layers_cfg["batch_norm"],
            dropout=self.reconstruction_layers_cfg["dropout"],
        )

        self.opter = opter_cls(self.parameters(), **opter_kwargs)

    def _build_resp_layers(self):
        layers = []
        in_channels = self.resp_shape[0]

        ### build layers from config
        for l_i, layer_config in enumerate(self.resp_layers_cfg["layers"]):
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

            if l_i < len(self.resp_layers_cfg["layers"]) - 1 and layer_type in ["fc", "conv", "deconv"]:
                ### add batch norm, activation, dropout
                if self.resp_layers_cfg["batch_norm"]:
                    if layer_type == "fc":
                        layers.append(nn.BatchNorm1d(out_channels))
                    else:
                        layers.append(nn.BatchNorm2d(out_channels))
                layers.append(self.resp_layers_cfg["act_fn"])
                if self.resp_layers_cfg["dropout"] > 0.0:
                    layers.append(nn.Dropout(self.resp_layers_cfg["dropout"]))
            elif l_i == len(self.resp_layers_cfg["layers"]) - 1:
                ### add output activation
                layers.append(self.resp_layers_cfg["out_act_fn"])

            in_channels = out_channels

        self.resp_layers = nn.Sequential(*layers)

    # @timeit
    def _get_resp_err_grad(self, x_hat, resp_target, data_key):
        resp_x_hat = self.encoder(normalize(x_hat, mean=0.441, std=0.254), data_key=data_key) # TODO: quick hack
        resp_loss = self.resp_loss_fn(resp_x_hat, resp_target)
        x_hat_grad = grad(resp_loss, x_hat, create_graph=False)[0]
        return x_hat_grad

    # @timeit
    def _preproc_grad(self, grad, preproc_factor=10.0):
        B, C, H, W = grad.shape
        preproc_threshold = np.exp(-preproc_factor)
        grad = grad.data # (B, C, H, W)

        grad_preproc = torch.zeros(B, C * 2, H, W, device=self.device)
        keep_grad_mask = (torch.abs(grad) >= preproc_threshold) # (B, C, H, W)
        
        ### if gradient is large enough, set to log(abs(grad)) / preproc_factor and sign(grad)
        grad_preproc[:, 0][keep_grad_mask.squeeze(1)] = (
            torch.log(torch.abs(grad[keep_grad_mask]) + 1e-8) / preproc_factor
        ).squeeze()
        grad_preproc[:, 1][keep_grad_mask.squeeze(1)] = (
            torch.sign(grad[keep_grad_mask]).squeeze()
        )

        ### if gradient is too small, set to -1 and exp(preproc_factor) * grad
        grad_preproc[:, 0][~keep_grad_mask.squeeze(1)] = -1
        grad_preproc[:, 1][~keep_grad_mask.squeeze(1)] = (
            float(np.exp(preproc_factor)) * grad[~keep_grad_mask]
        ).squeeze()
        
        return grad_preproc

    # @timeit
    def _init_reconstruction(self, resp):
        if self.reconstruction_init_method == "resp_layers":
            x_hat = self.resp_layers(resp)
        elif self.reconstruction_init_method == "random":
            x_hat = torch.rand(resp.shape[0], *self.stim_shape, device=self.device)
        elif self.reconstruction_init_method == "zero":
            x_hat = torch.zeros(resp.shape[0], *self.stim_shape, device=self.device)
        elif self.reconstruction_init_method == "identity":
            x_hat = resp
        return x_hat

    # @timeit
    def forward_single(self, x_hat, x_hat_grad, x_hat_t0):
        x = torch.cat([x_hat, x_hat_grad, x_hat_t0], dim=1) # (B, C*3, H, W)
        x = self.recon_layers(x)
        return x

    def forward(self, x, resp, stim, data_key, n_steps=1, train=False, x_hat_history_iters=None, neuron_coords=None, pupil_center=None):
        """ x received from a ReadIn or x == resp """
        assert not train or (train and stim is not None), \
            "You must provide stim during training."

        if train:
            self.train()
            unroll = self.unroll
        else:
            self.eval()
            unroll = 1

        ### init decoded imgs and hidden states
        x_hat = self._init_reconstruction(resp=x) # (B, C, H, W)
        if self.reconstruction_init_method in ("random", "zero", "identity"):
            x_hat.requires_grad = True
        else:
            x_hat.retain_grad()
        x_hat_t0 = x_hat.detach().clone()

        ### optimize decoded img
        unroll_loss = 0.
        history = {"x_hat_history": [], "loss": []}
        for step_i in range(1, n_steps + 1):
            ### get the gradient of the response error with respect to the current decoded image
            x_hat.requires_grad_(True)
            x_hat_grad = self._get_resp_err_grad(x_hat=x_hat, resp_target=resp, data_key=data_key).detach()
            if self.preproc_grad:
                x_hat_grad = self._preproc_grad(x_hat_grad)

            ### update the decoded image
            x_hat_delta = self.forward_single(
                x_hat=x_hat,
                x_hat_grad=x_hat_grad,
                x_hat_t0=x_hat_t0,
            )
            x_hat = F.sigmoid(x_hat + x_hat_delta)

            ### compute current reconstruction loss
            if stim is not None:
                curr_loss = self.stim_loss_fn(x_hat, stim)
                unroll_loss += curr_loss
            else:
                curr_loss = None

            ### finish unroll
            if step_i % unroll == 0:
                if train:
                    ### update decoder parameters
                    self.opter.zero_grad()
                    unroll_loss.backward()
                    self.opter.step()
                unroll_loss = 0.

                x_hat = x_hat.detach()
                x_hat.requires_grad = True

            ### log
            history["loss"].append(curr_loss.item() if curr_loss is not None else curr_loss)
            if x_hat_history_iters is not None and step_i in x_hat_history_iters:
                history["x_hat_history"].append(x_hat.detach().cpu())

        return x_hat, history

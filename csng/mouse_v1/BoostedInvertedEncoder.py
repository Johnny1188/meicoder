import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import GaussianBlur

from csng.losses import SSIMLoss
from csng.utils import normalize, crop
from csng.CNN_Decoder import CNN_Decoder


class BoostedInvertedEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 36, 64),
        stim_pred_init="zeros",
        opter_cls=torch.optim.SGD,
        opter_config={"lr": 1000},
        last_enc_inv_step=None,
        n_steps=500,
        resp_loss_fn=nn.MSELoss(reduction="sum"),
        stim_loss_fn=nn.MSELoss(),
        img_gauss_blur_config=None,
        img_grad_gauss_blur_config=None,
        img_normalize=False,
        refiner_cls=CNN_Decoder,
        refiner_config={
            "resp_shape": (1, 36, 64),
            "stim_shape": (1, 36, 64),
            "layers": [("conv", 64, 7, 1, 3), ("conv", 64, 5, 1, 2), ("conv", 64, 3, 1, 1)],
            "act_fn": nn.ReLU,
            "out_act_fn": nn.Identity,
            "dropout": 0.2,
            "batch_norm": True,
        },
        refiner_opter_cls=torch.optim.Adam,
        refiner_opter_config={"lr": 3e-4},
        refiner_crop_win=None,
        refiner_freq=10,
        refiner_loss_fn=nn.MSELoss(),
        eps_noise_after_refiner=None,
        device="cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.training = False
        self.encoder.eval()
        
        self.stim_pred_init = stim_pred_init
        self.img_dims = img_dims
        self.opter_cls = opter_cls
        self.opter_config = opter_config
        self.last_enc_inv_step = last_enc_inv_step
        self.n_steps = n_steps
        assert resp_loss_fn.reduction == "sum", "resp_loss_fn should have reduction='sum'"
        self.resp_loss_fn = resp_loss_fn
        self.stim_loss_fn = stim_loss_fn
        
        self.img_gauss_blur_config = img_gauss_blur_config
        self.img_grad_gauss_blur_config = img_grad_gauss_blur_config
        self.img_gauss_blur = None if img_gauss_blur_config is None else GaussianBlur(**img_gauss_blur_config)
        self.img_grad_gauss_blur = None if img_grad_gauss_blur_config is None else GaussianBlur(**img_grad_gauss_blur_config)
        self.img_normalize = img_normalize

        ### refiner network
        self.refiner_cls = refiner_cls
        self.refiner_config = refiner_config
        self.refiner = refiner_cls(**refiner_config)
        self.refiner_opter_cls = refiner_opter_cls
        self.refiner_opter_config = refiner_opter_config
        self.refiner_opter = refiner_opter_cls(self.refiner.parameters(), **refiner_opter_config)
        self.refiner_freq = refiner_freq
        self.refiner_crop_win = refiner_crop_win
        self.refiner_loss_fn = refiner_loss_fn
        self.eps_noise_after_refiner = eps_noise_after_refiner

        self.device = device

    def _init_x_hat(self, B):
        ### init decoded img
        if self.stim_pred_init == "zeros":
            x_hat = torch.zeros((B, *self.img_dims), requires_grad=True, device=self.device)
        elif self.stim_pred_init == "rand":
            x_hat = torch.rand((B, *self.img_dims), requires_grad=True, device=self.device)
        elif self.stim_pred_init == "randn":
            x_hat = torch.randn((B, *self.img_dims), requires_grad=True, device=self.device)
        else:
            raise ValueError(f"Unknown stim_pred_init: {self.stim_pred_init}")
        return x_hat

    def _enc_inv_step(self, x_hat, opter, resp, pupil_center, data_key):
        opter.zero_grad()
        if hasattr(self.encoder, "shifter") and self.encoder.shifter is not None:
            resp_pred = self.encoder(x_hat, data_key=data_key, pupil_center=pupil_center)
        else:
            resp_pred = self.encoder(x_hat, data_key=data_key)
        resp_loss = self.resp_loss_fn(resp_pred, resp) / resp.size(-1)
        resp_loss.backward()

        ### apply gaussian blur to gradients
        if self.img_grad_gauss_blur is not None:
            x_hat.grad = self.img_grad_gauss_blur(x_hat.grad)

        ### update
        opter.step()

        ### apply gaussian blur to image
        if self.img_gauss_blur is not None:
            x_hat = self.img_gauss_blur(x_hat)

        ### normalize image
        if self.img_normalize:
            x_hat = normalize(x_hat)

        return x_hat, opter, resp_pred, resp_loss.item()

    def fit_direct_iter(self, resp, stim_target, data_key, neuron_coords=None, pupil_center=None):
        assert resp.ndim > 1, "resp should be at least 2d (batch_dim, neurons_dim)"

        ### init decoded img: zeros -> 1-step encoder inversion -> preprocessing
        x_hat = self._init_x_hat(resp.size(0) if resp.ndim > 1 else 1)
        opter = self.opter_cls([x_hat], **self.opter_config)
        x_hat, opter, _, _ = self._enc_inv_step(x_hat, opter, resp, pupil_center, data_key)
        x_hat = x_hat.detach().requires_grad_(False)

        ### apply refiner network
        # sample random times
        t = torch.rand(resp.size(0), device=self.device)[:, None, None, None]
        # estimate the clean image: E[x_0 | x_t, t]
        x_t = (1 - t) * stim_target + t * x_hat
        x_t = crop(x_t, self.refiner_crop_win)
        x_hat_final = self.refiner(
            resp,
            data_key=data_key,
            neuron_coords=neuron_coords,
            pupil_center=pupil_center,
            # additional_core_channels=x_t,
            additional_core_inp={
                # "c1": torch.tensor(step_i / self.n_steps, device=self.device),
                "c1": t,
                "c2": None,
                "c3": x_t,
            }
            # additional_core_inp={
            #     "stim_pred": x_hat.detach().requires_grad_(False),
            #     "t": torch.tensor((step_i + 1) / self.n_steps, device=self.device),
            # },
        )

        ### update refiner network
        self.refiner_opter.zero_grad()
        loss = self.refiner_loss_fn(x_hat_final, stim_target, data_key=data_key, phase="train")
        loss.backward()
        self.refiner_opter.step()

        return loss.item()

    def forward(self, resp, data_key, train, stim_target=None, neuron_coords=None, pupil_center=None):
        assert not train or stim_target is not None, "stim_target should be provided if train=True"
        assert resp.ndim > 1, "resp should be at least 2d (batch_dim, neurons_dim)"
        assert not train, "Implementation now only supports train=False (i.e. inference). Please use fit_direct_iter for training."

        ### init decoded img
        x_hat = self._init_x_hat(resp.size(0) if resp.ndim > 1 else 1)

        ### optimize decoded img
        opter = self.opter_cls([x_hat], **self.opter_config)
        history = {"resp_loss": [], "stim_loss": [], "refiner_loss": [], "best": {"stim_loss": np.inf, "stim_pred": None}}
        for step_i in range(1, self.n_steps + 1):
            ### compute stim_loss
            if stim_target is not None:
                stim_loss = self.stim_loss_fn(x_hat.detach(), stim_target)
                history["stim_loss"].append(stim_loss.item())
                if stim_loss.item() < history["best"]["stim_loss"]:
                    history["best"]["stim_loss"] = stim_loss.item()
                    history["best"]["stim_pred"] = x_hat.detach().clone()

            ### encode and invert
            if self.last_enc_inv_step is None or step_i <= self.last_enc_inv_step:
                x_hat, opter, resp_pred, resp_loss = self._enc_inv_step(x_hat, opter, resp, pupil_center, data_key)

            ### apply refiner network
            if step_i % self.refiner_freq == 0:
                ### estimate E[x_0 | x_t, t]
                x_hat_final = self.refiner(
                    resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                    # additional_core_channels=crop(x_hat.detach().requires_grad_(False), self.refiner_crop_win),
                    additional_core_inp={
                        # "c1": torch.tensor(step_i / self.n_steps, device=self.device),
                        "c1": torch.tensor(1 - (step_i / self.n_steps), device=self.device),
                        "c2": None,
                        "c3": crop(x_hat.detach().requires_grad_(False), self.refiner_crop_win)
                    }
                    # additional_core_inp={
                    #     "stim_pred": x_hat.detach().requires_grad_(False),
                    #     "t": torch.tensor((step_i + 1) / self.n_steps, device=self.device),
                    # },
                )

                ### update x_hat
                delta = 1 / self.n_steps
                w_update = delta / (1. - ((step_i - 1) / self.n_steps))
                if x_hat.shape != x_hat_final.shape:
                    if step_i < self.last_enc_inv_step:
                        ### resize x_hat_final: place in the middle for the encoder
                        x_hat_final_for_enc = x_hat.detach().clone()
                        x_hat_final_for_enc[...,
                            (x_hat_final_for_enc.shape[-2] - self.refiner_crop_win[0])//2:(x_hat_final_for_enc.shape[-2] + self.refiner_crop_win[0])//2,
                            (x_hat_final_for_enc.shape[-1] - self.refiner_crop_win[1])//2:(x_hat_final_for_enc.shape[-1] + self.refiner_crop_win[1])//2
                        ] = x_hat_final
                        x_hat_final = x_hat_final_for_enc
                    else:
                        ### resize x_hat: crop
                        x_hat = crop(x_hat, self.refiner_crop_win)
                x_hat = (1 - w_update) * x_hat + w_update * x_hat_final
                if self.eps_noise_after_refiner is not None and self.eps_noise_after_refiner > 0.0:
                    x_hat = x_hat + (1 - (step_i / self.n_steps)) * self.eps_noise_after_refiner * torch.randn_like(x_hat)

                ### train refiner network
                if train:
                    self.refiner_opter.zero_grad()
                    loss = self.refiner_loss_fn(x_hat_final, stim_target, data_key=data_key, phase="train")
                    loss.backward()
                    self.refiner_opter.step()
                    history["refiner_loss"].append(loss.item())

            x_hat = x_hat.detach().requires_grad_(True)
            opter.param_groups[0]['params'] = [x_hat]

            ### log
            history["resp_loss"].append(resp_loss)

        return x_hat.detach(), resp_pred.detach(), history



class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _pad_concat(x1, x2):
        """ Pad x1 to match x2's shape and concatenate along the channel dimension. """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)

    def forward(self, x, skip):
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = self._pad_concat(x, skip)
        else:
            x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class EmbedConv(nn.Module):
    def __init__(self, input_dim, emb_dim, upsample_factor=1):
        super().__init__()
        '''
        generic one layer pointwise convolutional NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Conv2d(input_dim, emb_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(emb_dim, emb_dim, 1, 1, 0),
        ]
        if upsample_factor and upsample_factor > 1:
            layers.insert(0, nn.Upsample(scale_factor=upsample_factor))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        assert x.size(1) == self.input_dim, f"input dim {x.size(1)} does not match expected {self.input_dim}"
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(
        self,
        resp_channels,
        out_channels,
        channels=(128,256,512),
        bottleneck_avgpool=False,
        gn_mul_context=None,
        gn_add_context=None,
        c3_config=None,
        # gn_mul_context={
        #     "in_shape": (1000,),
        #     "embed_type": "fc",
        # },
        # gn_add_context={
        #     "in_shape": (1,),
        #     "embed_type": "fc",
        # },
        # c3_config={
        #     "in_shape": (1,22,36),
        #     "use_as": "input_concat", # "input_concat", "hidden_concat", "input"
        # },
    ):
        assert len(channels) == 3, "channels should be a tuple of length 3"
        super().__init__()

        self.resp_channels = resp_channels
        self.out_channels = out_channels
        self.channels = channels
        self.bottleneck_avgpool = bottleneck_avgpool

        ### encoder
        self.init_conv = ResidualConvBlock(c3_config["in_shape"][0], channels[0], is_res=True) \
            if c3_config is not None and c3_config["use_as"] == "input" \
            else ResidualConvBlock(resp_channels, channels[0], is_res=True)
        self.down1 = UnetDown(channels[0], channels[1])
        self.down2 = UnetDown(channels[1], channels[2])
        if bottleneck_avgpool:
            self.to_bneck = nn.Sequential(nn.AvgPool2d(2), nn.GELU())
        else:
            self.to_bneck = nn.Identity()

        ### context embeddings
        # c1
        self.gn_mul_c_config = gn_mul_context
        if self.gn_mul_c_config is not None:
            self.gn_mul = nn.ModuleList([
                self._get_embed_net(
                    embed_type=self.gn_mul_c_config["embed_type"],
                    in_shape=self.gn_mul_c_config["in_shape"][-1],
                    out_shape=channels[2],
                ),
                self._get_embed_net(
                    embed_type=self.gn_mul_c_config["embed_type"],
                    in_shape=self.gn_mul_c_config["in_shape"][-1],
                    out_shape=channels[1],
                ),
            ])
        # c2
        self.gn_add_c_config = gn_add_context
        if self.gn_add_c_config is not None:
            self.gn_add = nn.ModuleList([
                self._get_embed_net(
                    embed_type=self.gn_add_c_config["embed_type"],
                    in_shape=self.gn_add_c_config["in_shape"][-1],
                    out_shape=channels[2],
                ),
                self._get_embed_net(
                    embed_type=self.gn_add_c_config["embed_type"],
                    in_shape=self.gn_add_c_config["in_shape"][-1],
                    out_shape=channels[1],
                ),
            ])
        # c3
        self.c3_config = c3_config

        ### decoder
        up0 = [nn.GroupNorm(32, channels[2]), nn.ReLU()]
        if bottleneck_avgpool:
            up0.insert(0, nn.ConvTranspose2d(channels[2], channels[2], 2, 2))
        else:
            up0_in_channels = channels[2]
            if c3_config is not None and c3_config["use_as"] == "input":
                up0_in_channels += resp_channels
            up0.insert(0, nn.Conv2d(up0_in_channels, channels[2], 3, 1, 1))
        self.up0 = nn.Sequential(*up0)
        self.up1 = UnetUp(2 * channels[2], channels[1])
        self.up2 = UnetUp(2 * channels[1], channels[0])
        self.out = nn.Sequential(
            nn.Conv2d(2 * channels[0], channels[0], 3, 1, 1),
            nn.GroupNorm(32, channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], self.out_channels, 3, 1, 1),
        )

    def _get_embed_net(self, embed_type, in_shape, out_shape):
        if embed_type == "fc":
            return nn.Sequential(
                EmbedFC(in_shape, out_shape),
                nn.Unflatten(1, (out_shape, 1, 1))
            )
        elif embed_type == "conv":
            return EmbedConv(in_shape, out_shape)
        elif embed_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")

    @staticmethod
    def _pad_to(x, target_shape):
        diffY = target_shape[2] - x.size()[2]
        diffX = target_shape[3] - x.size()[3]
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

    def forward(self, resp, c1=None, c2=None, c3=None):
        ### x = imgs + responses, c1 and c2 = context embeddings for GN params
        assert self.c3_config is None or c3 is not None, \
            "c3 should be provided if c3_config is not None"

        ### prepare input
        if self.c3_config is not None:
            if self.c3_config["use_as"] == "input":
                x = c3
            elif self.c3_config["use_as"] == "hidden_concat":
                x = resp
            elif self.c3_config["use_as"] == "input_concat":
                x = torch.cat((c3, resp), 1)
            else:
                raise ValueError(f"Unknown use_as: {self.c3_config['use_as']}")
        else:
            x = resp

        ### down
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_bneck(down2)

        ### embed contexts for GN
        mul_emb1, mul_emb2, add_emb1, add_emb2 = 1, 1, 0, 0
        if self.gn_mul_c_config is not None:
            mul_emb1 = self.gn_mul[0](c1)
            mul_emb2 = self.gn_mul[1](c1)
        if self.gn_add_c_config is not None:
            add_emb1 = self.gn_add[0](c2)
            add_emb2 = self.gn_add[1](c2)

        ### c3 as "input" => resp as "hidden_concat"
        if self.c3_config is not None \
           and self.c3_config["use_as"] == "input":
            hiddenvec = torch.cat((hiddenvec, resp), 1)

        ### up
        up1 = self.up0(hiddenvec)
        up1 = self._pad_to(up1, down2.size())
        up2 = self.up1(mul_emb1 * up1 + add_emb1, down2)
        up2 = self._pad_to(up2, down1.size())
        up3 = self.up2(mul_emb2 * up2 + add_emb2, down1)
        up3 = self._pad_to(up3, x.size())
        out = self.out(torch.cat((up3, x), 1))

        return out

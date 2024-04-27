import os
import numpy as np
from scipy import signal
import dill
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import GaussianBlur


class InvertedEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 110, 110),
        stim_pred_init="zeros",
        opter_cls=torch.optim.SGD,
        opter_config={"lr": 0.1},
        n_steps=500,
        resp_loss_fn=F.mse_loss,
        stim_loss_fn=F.mse_loss,
        img_gauss_blur_config=None,
        img_gauss_blur_freq=1,
        img_grad_gauss_blur_config=None,
        img_grad_gauss_blur_freq=1,
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
        self.n_steps = n_steps
        self.resp_loss_fn = resp_loss_fn
        self.stim_loss_fn = stim_loss_fn
        
        self.img_gauss_blur_config = img_gauss_blur_config
        self.img_gauss_blur_freq = img_gauss_blur_freq
        self.img_gauss_blur = None if img_gauss_blur_config is None else GaussianBlur(**img_gauss_blur_config)
        self.img_grad_gauss_blur_config = img_grad_gauss_blur_config
        self.img_grad_gauss_blur_freq = img_grad_gauss_blur_freq
        self.img_grad_gauss_blur = None if img_grad_gauss_blur_config is None else GaussianBlur(**img_grad_gauss_blur_config)

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

    def forward(self, resp_target, stim_target=None, additional_encoder_inp=None, ckpt_config=None):
        assert resp_target.ndim > 1, "resp_target should be at least 2d (batch_dim, neurons_dim)"

        ### init decoded img
        x_hat = self._init_x_hat(resp_target.size(0) if resp_target.ndim > 1 else 1)

        ### optimize decoded img
        opter = self.opter_cls([x_hat], **self.opter_config)
        history = {"resp_loss": [], "stim_loss": [], "best": {"stim_loss": np.inf, "stim_pred": None}}
        for step_i in range(self.n_steps):
            opter.zero_grad()

            resp_pred = self.encoder(x_hat) if additional_encoder_inp is None else self.encoder(x_hat, **additional_encoder_inp)
            resp_loss = self.resp_loss_fn(resp_pred, resp_target)
            resp_loss.backward()

            ### apply gaussian blur to gradients
            if self.img_grad_gauss_blur is not None and step_i % self.img_grad_gauss_blur_freq == 0:
                x_hat.grad = self.img_grad_gauss_blur(x_hat.grad)

            ### update
            opter.step()
            if stim_target is not None:
                stim_loss = self.stim_loss_fn(x_hat.detach(), stim_target)
                history["stim_loss"].append(stim_loss.item())
                if stim_loss.item() < history["best"]["stim_loss"]:
                    history["best"]["stim_loss"] = stim_loss.item()
                    history["best"]["stim_pred"] = x_hat.detach().clone()

            ### apply gaussian blur to image
            if self.img_gauss_blur is not None and step_i % self.img_gauss_blur_freq == 0:
                with torch.no_grad():
                    x_hat.data = self.img_gauss_blur(x_hat)

            ### log
            history["resp_loss"].append(resp_loss.item())

            ### ckpt
            if ckpt_config is not None and step_i % ckpt_config["ckpt_freq"] == 0:
                curr_ckpt_dir = os.path.join(ckpt_config["ckpt_dir"], str(step_i))
                os.makedirs(curr_ckpt_dir)
                torch.save({
                    "reconstruction": x_hat,
                    "history": history,
                    "opter_state": opter.state_dict(),
                }, os.path.join(curr_ckpt_dir, "ckpt.pt"), pickle_module=dill)
                if ckpt_config.get("plot_fn", None) is not None:
                    ckpt_config["plot_fn"](target=stim_target, pred=x_hat, save_to=os.path.join(curr_ckpt_dir, f"stim_pred.png"))

        return x_hat.detach(), resp_pred.detach(), history


# class GaussianBlur:
#     """
#     Source: https://github.com/sinzlab/energy-guided-diffusion/blob/main/scripts/reconstruct_gd.py
#     """

#     """Blur an image with a Gaussian window.
#     Arguments:
#         sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
#         decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
#             (iteration - 1)`. Ignored if None.
#         truncate (float): Gaussian window is truncated after this number of standard
#             deviations to each side. Size of kernel = 8 * sigma + 1
#         pad_mode (string): Mode for the padding used for the blurring. Valid values are:
#             'constant', 'reflect' and 'replicate'
#         mei_only (True/False): for transparent mei, if True, no Gaussian blur for transparent channel:
#             default should be False (also for non transparent case)
#     """

#     def __init__(
#         self, sigma, decay_factor=None, truncate=4, pad_mode="reflect",
#     ):
#         self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
#         self.decay_factor = decay_factor
#         self.truncate = truncate
#         self.pad_mode = pad_mode

#     def __call__(self, x, iteration=None):
#         ### update sigma if needed
#         if self.decay_factor is None:
#             sigma = self.sigma
#         else:
#             sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

#         ### define 1-d kernels to use for blurring
#         y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
#         y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
#         x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
#         x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
#         y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
#         x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

#         ### blur
#         c = x.shape[1]
#         padded_x = F.pad(
#             x,
#             pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
#             mode=self.pad_mode,
#         )
#         blurred_x = F.conv2d(
#             padded_x,
#             y_gaussian.repeat(c, 1, 1)[..., None],
#             groups=c,
#         )
#         blurred_x = F.conv2d(
#             blurred_x, x_gaussian.repeat(c, 1, 1, 1), groups=c
#         )
#         final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
#         return final_x

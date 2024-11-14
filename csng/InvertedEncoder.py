import os
import numpy as np
import dill
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import GaussianBlur
import featurevis
from featurevis import ops
from featurevis import utils as fvutils


class InvertedEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 110, 110),
        stim_pred_init="zeros",
        opter_cls=torch.optim.SGD,
        opter_config={"lr": 0.1},
        n_steps=500,
        resp_loss_fn=lambda resp_pred, resp_target: F.mse_loss(resp_pred, resp_target, reduction="none").mean(-1).sum(),
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

        self.resp_pred = None
        self.history = None

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

    def forward(self, resp, data_key=None, neuron_coords=None, pupil_center=None, ckpt_config=None, stim_target=None):
        assert resp.ndim > 1, "resp should be at least 2d (batch_dim, neurons_dim)"

        ### init decoded img
        x_hat = self._init_x_hat(resp.size(0) if resp.ndim > 1 else 1)

        ### optimize decoded img
        opter = self.opter_cls([x_hat], **self.opter_config)
        history = {"resp_loss": [], "stim_loss": [], "best": {"stim_loss": np.inf, "stim_pred": None}}
        for step_i in range(self.n_steps):
            opter.zero_grad()

            resp_pred = self.encoder(x_hat, data_key=data_key, pupil_center=pupil_center)
            resp_loss = self.resp_loss_fn(resp_pred, resp)
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

        self.resp_pred = resp_pred.detach()
        self.history = history
        return x_hat.detach()


class InvertedEncoderBrainreader(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 36, 64),
        stim_pred_init="randn",
        lr=100,
        n_steps=500,
        img_grad_gauss_blur_sigma=0,
        jitter=None,
        mse_reduction="per_sample_mean_sum",
        device="cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.training = False
        self.encoder.eval()

        self.mse_reduction = mse_reduction
        self.stim_pred_init = stim_pred_init
        self.img_dims = img_dims
        self.lr = lr
        self.n_steps = n_steps

        self.img_grad_gauss_blur_sigma = img_grad_gauss_blur_sigma
        self.jitter = jitter

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

    def forward(self, resp, data_key=None, neuron_coords=None, pupil_center=None):
        assert resp.ndim > 1, "resp should be at least 2d (batch_dim, neurons_dim)"

        ### run separately for each image
        recons = []
        # Set up initial image
        # torch.manual_seed(0)
        initial_image = self._init_x_hat(resp.shape[0])

        # Set up optimization function
        neural_resp = torch.as_tensor(resp, dtype=torch.float32, device=self.device)
        similarity = fvutils.Compose([ops.MSE(neural_resp, reduction=self.mse_reduction), ops.MultiplyBy(-1)])
        encoder_pred = lambda x: self.encoder(x, data_key=data_key, pupil_center=pupil_center)
        obj_function = fvutils.Compose([encoder_pred, similarity])

        # Optimize
        jitter = ops.Jitter(self.jitter) if self.jitter is not None and self.jitter != 0 else None
        blur = (ops.GaussianBlur(self.img_grad_gauss_blur_sigma)
                if self.img_grad_gauss_blur_sigma != 0 else None)
        recon, fevals, _ = featurevis.gradient_ascent(
            obj_function,
            initial_image,
            step_size=self.lr,
            num_iterations=self.n_steps,
            transform=jitter,
            gradient_f=blur,
            regularization=None,
            post_update=ops.ChangeStd(1),
            print_iters=None,
        )
        recons.append(recon.detach())

        return torch.cat(recons, dim=0)

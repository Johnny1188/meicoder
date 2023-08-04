from scipy import signal
import torch
import torch.nn.functional as F
from torch import nn


class InvertedEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        img_dims=(1, 110, 110),
        lr=1e-1,
        n_steps=500,
        resp_loss_fn=F.mse_loss,
        img_gaussian_blur_sigma=None,
        img_grad_gaussian_blur_sigma=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.training = False
        self.encoder.eval()
        self.img_dims = img_dims
        self.lr = lr
        self.n_steps = n_steps
        self.resp_loss_fn = resp_loss_fn
        self.img_gaussian_blur = None if img_gaussian_blur_sigma is None else GaussianBlur(sigma=img_gaussian_blur_sigma)
        self.img_grad_gaussian_blur = None if img_grad_gaussian_blur_sigma is None else GaussianBlur(sigma=img_grad_gaussian_blur_sigma)

    def _init_x_hat(self, resp_target):
        ### init decoded img
        x_hat = torch.zeros((resp_target.shape[0], *self.img_dims), requires_grad=True, device=resp_target.device)
        return x_hat

    def forward(self, resp_target, x_hat_history_iters=None):
        assert resp_target.ndim > 1, "resp_target should be at least 2d (batch_dim, neurons_dim)"

        ### init decoded img
        x_hat = self._init_x_hat(resp_target)

        ### optimize decoded img
        opter = torch.optim.Adam([x_hat], lr=self.lr)
        loss_history = []
        x_hat_history = [] if x_hat_history_iters is not None else None
        for step_i in range(self.n_steps):
            opter.zero_grad()
            resp_pred = self.encoder(x_hat)
            loss = self.resp_loss_fn(resp_pred, resp_target)
            loss.backward()

            ### apply gaussian blur to gradients
            if self.img_grad_gaussian_blur is not None:
                with torch.no_grad():
                    x_hat.grad = self.img_grad_gaussian_blur(x_hat.grad)

            opter.step()

            ### apply gaussian blur to image
            if self.img_gaussian_blur is not None:
                with torch.no_grad():
                    x_hat = self.img_gaussian_blur(x_hat)

            ### log
            loss_history.append(loss.item())
            if x_hat_history is not None and step_i in x_hat_history_iters:
                x_hat_history.append(x_hat.detach().cpu())

        return x_hat.detach(), resp_pred.detach(), loss_history, x_hat_history


class GaussianBlur:
    """
    Source: https://github.com/sinzlab/energy-guided-diffusion/blob/main/scripts/reconstruct_gd.py
    """

    """Blur an image with a Gaussian window.
    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
        mei_only (True/False): for transparent mei, if True, no Gaussian blur for transparent channel:
            default should be False (also for non transparent case)
    """

    def __init__(
        self, sigma, decay_factor=None, truncate=4, pad_mode="reflect",
    ):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode

    def __call__(self, x, iteration=None):
        ### update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        ### define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        ### blur
        c = x.shape[1]
        padded_x = F.pad(
            x,
            pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
            mode=self.pad_mode,
        )
        blurred_x = F.conv2d(
            padded_x,
            y_gaussian.repeat(c, 1, 1)[..., None],
            groups=c,
        )
        blurred_x = F.conv2d(
            blurred_x, x_gaussian.repeat(c, 1, 1, 1), groups=c
        )
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        return final_x

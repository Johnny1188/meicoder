from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
import torchmetrics
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models.feature_extraction import create_feature_extractor

from csng.utils.data import standardize, normalize, crop


def get_metrics(inp_zscored, crop_win=None, reduction="mean", device="cpu"):
    metrics = {
        "SSIM": Loss(config=dict(
            loss_fn=SSIM(reduction=reduction),
            window=crop_win,
            standardize=inp_zscored,
        )),
        "Log SSIML": Loss(config=dict(
            loss_fn=SSIMLoss(
                log_loss=True,
                inp_normalized=inp_zscored,
                inp_standardized=not inp_zscored,
                reduction=reduction,
            ),
            window=crop_win,
        )),
        "SSIML": Loss(config=dict(
            loss_fn=SSIMLoss(
                log_loss=False,
                inp_normalized=inp_zscored,
                inp_standardized=not inp_zscored,
                reduction=reduction,
            ),
            window=crop_win,
        )),
        "PixCorr": Loss(config=dict(
            loss_fn=PixCorr(reduction=reduction),
            window=crop_win,
        )),
        "PixCorr Loss": Loss(config=dict(
            loss_fn=HigherBetterMetricToLossWrapper(metric=PixCorr(reduction=reduction), metric_range=(0, 1)),
            window=crop_win,
        )),
        "Alex(2)": Loss(config=dict(
            loss_fn=TwoWayAlexNet(
                inp_zscored=inp_zscored,
                feature_layers=["features.4"],
                avg_across_layers=True,
                reduction=reduction,
                device=device,
            ),
            window=crop_win,
        )),
        "Alex(2) Loss": Loss(config=dict(
            loss_fn=HigherBetterMetricToLossWrapper(
                metric=TwoWayAlexNet(
                    inp_zscored=inp_zscored,
                    feature_layers=["features.4"],
                    avg_across_layers=True,
                    reduction=reduction,
                    device=device,
                ),
                metric_range=(0, 1),
            ),
            window=crop_win,
        )),
        "Alex(5)": Loss(config=dict(
            loss_fn=TwoWayAlexNet(
                inp_zscored=inp_zscored,
                feature_layers=["features.11"],
                avg_across_layers=True,
                reduction=reduction,
                device=device,
            ),
            window=crop_win,
        )),
        "Alex(5) Loss": Loss(config=dict(
            loss_fn=HigherBetterMetricToLossWrapper(
                metric=TwoWayAlexNet(
                    inp_zscored=inp_zscored,
                    feature_layers=["features.11"],
                    avg_across_layers=True,
                    reduction=reduction,
                    device=device,
                ),
                metric_range=(0, 1),
            ),
            window=crop_win,
        )),
        "PL": Loss(config=dict(
            loss_fn=VGGPerceptualLoss(
                resize=False,
                device=device,
                reduction=reduction,
            ),
            window=crop_win,
            standardize=inp_zscored,
        )),
        "FFL": Loss(config=dict(
            loss_fn=FFL(loss_weight=1, alpha=1.0),
            window=crop_win,
            standardize=inp_zscored,
        )),
        # "MSE": Loss(config=dict(
        #     loss_fn=lambda x_hat, x: F.mse_loss(
        #         standardize(crop(x_hat, crop_win)) if inp_zscored else crop(x_hat, crop_win),
        #         standardize(crop(x, crop_win)) if inp_zscored else crop(x, crop_win),
        #         reduction="none",
        #     ).mean((1,2,3)).sum(),
        #     window=crop_win,
        # )),
        "MSE": Loss(config=dict(
            loss_fn=MSELoss(
                window=crop_win,
                minmax_normalize=inp_zscored,
                reduction=reduction,
            ),
        )),
        "MSE w/out min-max normalization": Loss(config=dict(
            loss_fn=MSELoss(
                window=crop_win,
                minmax_normalize=False,
                reduction=reduction,
            ),
        )),
        # "MSE-no-standardization": Loss(config=dict(
        #     loss_fn=lambda x_hat, x: F.mse_loss(
        #         crop(x_hat, crop_win),
        #         crop(x, crop_win),
        #         reduction="none",
        #     ).mean((1,2,3)).sum(),
        #     window=crop_win,
        # )),
        # "MAE": Loss(config=dict(
        #     loss_fn=lambda x_hat, x: F.l1_loss(
        #         standardize(crop(x_hat, crop_win)) if inp_zscored else crop(x_hat, crop_win),
        #         standardize(crop(x, crop_win)) if inp_zscored else crop(x, crop_win),
        #         reduction="none",
        #     ).mean((1,2,3)).sum(),
        #     window=crop_win,
        # )),
        "MAE": Loss(config=dict(
            loss_fn=MAELoss(
                window=crop_win,
                minmax_normalize=inp_zscored,
                reduction=reduction,
            ),
        )),
        "MAE w/out min-max normalization": Loss(config=dict(
            loss_fn=MAELoss(
                window=crop_win,
                minmax_normalize=False,
                reduction=reduction,
            ),
        )),
    }
    metrics["SSIML-PL"] = Loss(config=dict(
        loss_fn=lambda y_hat, y, **kwargs: metrics["SSIML"](y_hat, y, **kwargs) + metrics["PL"](y_hat, y, **kwargs).to(y_hat.device),
        window=crop_win,
    ))

    return metrics


class Loss:
    def __init__(self, config, update_loss_fn_kwargs_with_items=None, model=None):
        self.model = model
        self.update_loss_fn_kwargs_with_items = update_loss_fn_kwargs_with_items
        self.loss_fn = config["loss_fn"]() if type(config["loss_fn"]) == type else config["loss_fn"]

        self.l1_reg_mul = config.get("l1_reg_mul", 0.)
        self.l2_reg_mul = config.get("l2_reg_mul", 0.)
        self.standardize = config.get("standardize", False)
        self.normalize = config.get("normalize", False)
        self.window = config.get("window", None)

        ### noise regularization
        self.noise_reg = config.get("noise_reg", None)
        self.noise_reg_loss_fn = config.get("noise_reg_loss_fn", None)
        self.noise_reg_mul = config.get("noise_reg_mul", 0.)

        ### noise data augmentation
        self.noise_data_aug = config.get("noise_data_aug", None)
        self.noise_data_aug_loss_fn = config.get("noise_data_aug_loss_fn", None)
        self.noise_data_aug_mul = config.get("noise_data_aug_mul", 0.)

        ### brain distance
        self.brain_distance_loss_fn = None
        self.brain_distance_mul = config.get("brain_distance_mul", None)
        if self.brain_distance_mul is not None and self.brain_distance_mul > 0:
            self.brain_distance_loss_fn = BrainDistance(**config["brain_distance_config"])

    def _brain_distance_aux_loss_fn(self, stim_pred, stim, resp, data_key, neuron_coords=None, pupil_center=None):
        if self.brain_distance_loss_fn is not None:
            return self.brain_distance_loss_fn(
                pred=stim_pred,
                target=stim,
                resp=resp,
                data_key=data_key,
                neuron_coords=neuron_coords,
                pupil_center=pupil_center,
            )
        return 0.

    def _noise_reg_loss_fn(self, stim_pred, resp, data_key, neuron_coords=None, pupil_center=None):
        if self.noise_reg is not None:
            noised_resp = self.noise_reg[data_key]._add_noise(responses=resp)
            return self.noise_reg_loss_fn(
                self.model(noised_resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center),
                stim_pred.detach(), # observed better results than w/out detaching
            )
        return 0.

    def _noise_data_aug_loss_fn(self, stim, resp, data_key, neuron_coords=None, pupil_center=None):
        if self.noise_data_aug is not None:
            noised_resp = self.noise_data_aug[data_key]._add_noise(responses=resp)
            return self.noise_data_aug_loss_fn(
                self.model(noised_resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center),
                stim,
            )
        return 0.

    def __call__(self, stim_pred, stim, resp=None, data_key=None, neuron_coords=None, pupil_center=None, additional_core_inp=None, sum_over_samples=False, phase="train"):
        assert phase in ("train", "val"), f"phase {phase} not recognized"

        ### crop only window
        if self.window is not None:
            stim_pred = crop(stim_pred, win=self.window)
            stim = crop(stim, win=self.window)

        ### standardize and/or normalize
        if self.normalize:
            stim_pred = normalize(stim_pred)
            stim = normalize(stim)
        if self.standardize:
            stim_pred = standardize(stim_pred)
            stim = standardize(stim)

        ### compute loss
        loss_fn = self.loss_fn
        if type(loss_fn) == dict: # different loss functions for different data keys
            loss_fn = loss_fn[data_key]

        loss_fn_kwargs = {}
        if loss_fn.__class__ == Loss:
            loss_fn_kwargs = dict(resp=resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center, additional_core_inp=additional_core_inp, phase=phase)
        if self.update_loss_fn_kwargs_with_items is not None:
            if "resp" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["resp"] = resp
            if "data_key" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["data_key"] = data_key
            if "neuron_coords" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["neuron_coords"] = neuron_coords
            if "pupil_center" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["pupil_center"] = pupil_center
            if "additional_core_inp" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["additional_core_inp"] = additional_core_inp
            if "phase" in self.update_loss_fn_kwargs_with_items: loss_fn_kwargs["phase"] = phase

        if phase == "val":
            if hasattr(loss_fn, "reduction"):
                if sum_over_samples:
                    before_red = loss_fn.reduction
                    loss_fn.reduction = "none"
                    loss = loss_fn(stim_pred, stim, **loss_fn_kwargs).sum(0).mean()
                    loss_fn.reduction = before_red
                else:
                    loss = loss_fn(stim_pred, stim, **loss_fn_kwargs).mean()
            else:
                if sum_over_samples:
                    loss = sum(loss_fn(stim_pred[i][None,:,:,:], stim[i][None,:,:,:], **loss_fn_kwargs) for i in range(stim_pred.shape[0]))
                else:
                    loss = loss_fn(stim_pred, stim, **loss_fn_kwargs).mean()
        else:
            loss = loss_fn(stim_pred, stim, **loss_fn_kwargs)

        ### L1 regularization
        if self.l1_reg_mul != 0 and phase == "train":
            l1_reg = sum(p.abs().sum() for n, p in self.model.named_parameters() 
                         if p.requires_grad and "weight" in n and (data_key is None or data_key in n))
            loss += self.l1_reg_mul * l1_reg

        ### L2 regularization
        if self.l2_reg_mul != 0 and phase == "train":
            l2_reg = sum(p.pow(2.0).sum() for n, p in self.model.named_parameters() 
                         if p.requires_grad and "weight" in n and (data_key is None or data_key in n))
            loss += self.l2_reg_mul * l2_reg

        ### noise regularization
        if self.noise_reg is not None and phase == "train":
            loss += self.noise_reg_mul * self._noise_reg_loss_fn(stim_pred=stim_pred, resp=resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center)

        ### noise data augmentation
        if self.noise_data_aug is not None and phase == "train":
            loss += self.noise_data_aug_mul * self._noise_data_aug_loss_fn(stim=stim, resp=resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center)

        ### brain distance auxiliary loss
        if self.brain_distance_loss_fn is not None and phase == "train":
            loss += self.brain_distance_mul * self._brain_distance_aux_loss_fn(stim_pred=stim_pred, stim=stim, resp=resp, data_key=data_key, neuron_coords=neuron_coords, pupil_center=pupil_center)

        return loss


class HigherBetterMetricToLossWrapper:
    def __init__(self, metric, metric_range=None):
        assert metric_range is None or type(metric_range) in (list, tuple), "metric_range should be a list or tuple"
        assert metric_range is None or len(metric_range) == 2, "metric_range should have 2 elements"
        assert metric_range is None or metric_range[0] < metric_range[1], "metric_range should be increasing"

        self.metric = metric
        self.metric_range = metric_range

    def __call__(self, *args, **kwargs):
        if self.metric_range is None:
            return -self.metric(*args, **kwargs)
        else: # squash to [0, 1]
            metric_val = self.metric(*args, **kwargs)
            return 1 - (metric_val - self.metric_range[0]) / (self.metric_range[1] - self.metric_range[0])


### source of SSIM-based losses and associated utility functions: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            print(f"[WARNING] Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}")

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = False,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        inp_normalized=True,
        inp_standardized=False,
        log_loss=False,
        window=None, # (x1, x2, y1, y2)
        size_average=False,
        win_size=11,
        win_sigma=1.5,
        channel=1,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
        reduction="mean",
    ):
        assert reduction in ["mean", "sum", "none"], f"Reduction {reduction} not recognized. Use 'mean', 'sum' or 'none'."
        assert (inp_normalized and not inp_standardized) or (
            not inp_normalized and inp_standardized
        ), "Input should be either normalized or standardized."

        super().__init__()
        self.inp_normalized = inp_normalized
        self.inp_standardized = inp_standardized
        self.log_loss = log_loss
        self.window = window
        self.size_average = size_average
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.channel = channel
        self.spatial_dims = spatial_dims
        self.K = K
        self.reduction = reduction
        self.nonnegative_ssim = nonnegative_ssim

        self.ssim = SSIM(
            data_range=1.0,
            # size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            # channel=channel,
            # spatial_dims=spatial_dims,
            K=K,
            nonnegative=nonnegative_ssim,
        )
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        ### crop only window
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if self.inp_normalized:
            pred = standardize(pred)
            target = standardize(target)

        assert torch.all(pred >= -1e-5) and torch.all(pred <= 1 + 1e-5), "Predictions should be in the [0, 1] range."
        assert torch.all(target >= 0) and torch.all(target <= 1), "Targets should be in the [0, 1] range."
        ssim_val = self.ssim(pred, target) # (B,)
        assert ssim_val.ndim == 1 and ssim_val.size(0) == pred.size(0), \
            "Incorrect dimensions encountered in computing the SSIM value."

        if not self.nonnegative_ssim:
            # ssim value is in range [-1, 1] - shift to [0, 1]
            ssim_val = (ssim_val + 1) / 2

        if self.log_loss:
            loss = -torch.log(ssim_val + 1e-6)
        else:
            loss = (1 - ssim_val)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass

        return loss


class MultiSSIMLoss(torch.nn.Module):
    """ Multi-SSIM Loss: combines SSIM losses with different hyperparameters. """
    def __init__(
        self,
        inp_normalized: bool = True,
        inp_standardized: bool = False,
        log_loss: bool = False,
        window=None, # (x1, x2, y1, y2)
        size_average: bool = False,
        win_sizes: List[int] = [5, 7, 9, 11, 13],
        win_sigmas: List[float] = [0.8, 1.2, 1.5, 1.8, 2.0],
        channel: int = 1,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
        reduction: str = "mean",
        weights: Optional[List[float]] = None,
    ) -> None:
        assert (inp_normalized and not inp_standardized) or (
            not inp_normalized and inp_standardized
        ), "Input should be either normalized or standardized."

        super().__init__()
        self.inp_normalized = inp_normalized
        self.inp_standardized = inp_standardized
        self.log_loss = log_loss
        self.window = window
        self.size_average = size_average
        self.win_sizes = win_sizes
        self.win_sigmas = win_sigmas
        self.channel = channel
        self.spatial_dims = spatial_dims
        self.K = K
        self.reduction = reduction
        self.weights = weights
        self.nonnegative_ssim = nonnegative_ssim

        ### create all combinations of ssim losses
        ssim_combinations = [(ws, sig) for ws in win_sizes for sig in win_sigmas]
        self.ssim_losses = dict()
        for w_cfg in ssim_combinations:
            self.ssim_losses[w_cfg] = SSIM(
                data_range=1.0,
                size_average=size_average,
                win_size=w_cfg[0],
                win_sigma=w_cfg[1],
                channel=channel,
                spatial_dims=spatial_dims,
                K=K,
                nonnegative_ssim=nonnegative_ssim,
            )


    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        r""" function for computing ssim loss
        Args:
            X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Returns:
            torch.Tensor: ssim results
        """

        ### loss wrt window
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if self.inp_normalized:
            pred = standardize(pred)
            target = standardize(target)

        ### compute ssim losses
        ssim_vals = []
        for ssim_loss in self.ssim_losses.values():
            ssim_vals.append(ssim_loss(pred, target))

        ### combine ssim losses
        ssim_vals = torch.stack(ssim_vals, dim=1) # (B, N)
        ssim_val = ssim_vals.mean(dim=1) # (B,)
        
        if not self.nonnegative_ssim:
            # ssim value is in range [-1, 1] - shift to [0, 1]
            ssim_val = (ssim_val + 1) / 2
        if self.log_loss:
            loss = -torch.log(ssim_val + 1e-6)
        else:
            loss = (1 - ssim_val)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass

        return loss


class MS_SSIMLoss(torch.nn.Module):
    r""" class for ms-ssim loss
    Args:
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        channel: (int, optional): input channels (default: 3)
        spatial_dims: (int, optional): spatial dims (default: 2)
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    """

    def __init__(
        self,
        inp_normalized: bool = True,
        inp_standardized: bool = False,
        log_loss: bool = False,
        window=None, # (x1, x2, y1, y2)
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 1,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        assert (inp_normalized and not inp_standardized) or (
            not inp_normalized and inp_standardized
        ), "Input should be either normalized or standardized."

        super().__init__()
        self.inp_normalized = inp_normalized
        self.inp_standardized = inp_standardized
        self.log_loss = log_loss
        self.window = window
        self.size_average = size_average
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.channel = channel
        self.spatial_dims = spatial_dims
        self.weights = weights
        self.K = K

        self.ms_ssim = MS_SSIM(
            data_range=1.0,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=channel,
            spatial_dims=spatial_dims,
            weights=weights,
            K=K,
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        r""" function for computing ms-ssim loss
        Args:
            X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Returns:
            torch.Tensor: ms-ssim results
        """

        ### loss wrt window
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if self.inp_normalized:
            pred = standardize(pred)
            target = standardize(target)

        ms_ssim_val = self.ms_ssim(pred, target)

        if self.log_loss:
            loss = -torch.log(ms_ssim_val + 1e-6)
        else:
            loss = 1 - ms_ssim_val

        return loss


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=1,
        # size_average=False,
        win_size=11,
        win_sigma=1.5,
        # channel=1,
        # spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative=False,
        reduction="none",
    ):
        assert reduction in ["mean", "sum", "none"], f"Invalid reduction: {reduction}"
        super().__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        # self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        # self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative = nonnegative
        self.reduction = reduction

    def _ssim(self, x, y):
        # return ssim(
        #     X=x,
        #     Y=y,
        #     data_range=self.data_range,
        #     size_average=self.size_average,
        #     win_size=self.win_size,
        #     win_sigma=self.win_sigma,
        #     win=self.win,
        #     K=self.K,
        #     nonnegative_ssim=self.nonnegative,
        # )
        return torchmetrics.functional.image.structural_similarity_index_measure(
            preds=x,
            target=y,
            gaussian_kernel=True,
            sigma=self.win_sigma,
            kernel_size=self.win_size,
            reduction="none", # done manually
            data_range=self.data_range,
            k1=self.K[0],
            k2=self.K[1],
            return_full_image=False,
            return_contrast_sensitivity=False,
        )

    def forward(self, x, y):
        ssim_val = self._ssim(x, y)

        if self.nonnegative:
            ssim_val = F.relu(ssim_val)

        if self.reduction == "mean":
            ssim_val = ssim_val.mean()
        elif self.reduction == "sum":
            ssim_val = ssim_val.sum()
        elif self.reduction == "none":
            pass

        return ssim_val


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


class MSELoss(torch.nn.Module):
    def __init__(self, window=None, minmax_normalize=False, reduction="mean"):
        super().__init__()
        self.window = window
        self.minmax_normalize = minmax_normalize
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if self.minmax_normalize:
            pred = standardize(pred)
            target = standardize(target)

        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class MAELoss(torch.nn.Module):
    def __init__(self, window=None, minmax_normalize=False, reduction="mean"):
        super().__init__()
        self.window = window
        self.minmax_normalize = minmax_normalize
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if self.minmax_normalize:
            pred = standardize(pred)
            target = standardize(target)

        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss


### Perceptual Loss
class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        inp_standardized: bool = False,
        window=None, # (x1, x2, y1, y2)
        resize: bool = True,
        reduction: str = "mean",
        device: str = "cuda",
    ):
        assert reduction in ["mean", "sum", "none"], f"Reduction {reduction} not recognized. Use 'mean', 'sum' or 'none'."
        super().__init__()
        self.inp_standardized = inp_standardized
        self.window = window
        self.vgg_loss = VGGPerceptualLoss(resize=resizem, reduction=reduction).to(device)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.window is not None:
            pred = crop(pred, win=self.window)
            target = crop(target, win=self.window)

        if not self.inp_standardized:
            pred = standardize(pred)
            target = standardize(target)

        return self.vgg_loss(pred, target)


class VGGPerceptualLoss(torch.nn.Module):
    """ Modified from: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49 """
    def __init__(self, resize=False, mean_across_layers=True, mul_factor=1/4, reduction="per_sample_mean_sum", device="cuda"):
        assert reduction in ["none", "per_sample_mean", "per_sample_mean_sum", "mean"], f"Invalid reduction: {reduction}"
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.reduction = reduction
        self.mul_factor = mul_factor
        self.mean_across_layers = mean_across_layers
        self.device = device
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _loss_fn(self, y_hat, y):
        if self.reduction == "none" or self.reduction == "per_sample_mean":
            loss = torch.nn.functional.l1_loss(y_hat, y, reduction="none")
            loss = loss.mean(dim=[1, 2, 3])
        elif self.reduction == "per_sample_mean_sum":
            loss = torch.nn.functional.l1_loss(y_hat, y, reduction="none")
            loss = loss.mean(dim=[1, 2, 3]).sum()
        elif self.reduction == "mean":
            loss = torch.nn.functional.l1_loss(y_hat, y, reduction="mean")
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        return loss

    def forward(self, inp, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        assert inp.min() >= 0.0 and inp.max() <= 1.0, "Input should be normalized to [0, 1] range."

        if inp.shape[1] != 3:
            inp = inp.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        inp = (inp.to(self.device) - self.mean) / self.std
        target = (target.to(self.device) - self.mean) / self.std
        if self.resize:
            inp = self.transform(inp, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = 0.0
        n_ls = 0
        x = inp
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += self._loss_fn(x, y)
                n_ls += 1
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += self._loss_fn(gram_x, gram_y)
                n_ls += 1

        if self.mean_across_layers:
            if n_ls > 0:
                loss = loss / n_ls

        loss = loss * self.mul_factor

        return loss


class FID:
    def __init__(self, inp_standardized=False, device="cpu"):
        ### note  inp_standardized == True <=> inputs in [0, 1] (different naming)
        self.inp_standardized = inp_standardized
        self.fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)

    @torch.no_grad()
    def __call__(self, pred_imgs, gt_imgs):
        assert pred_imgs.shape == gt_imgs.shape, \
            f"Shapes are not the same! pred_imgs have shape {pred_imgs.shape} while gt_imgs have shape {gt_imgs.shape}."

        ### standardize to [0, 1] and then to [0, 255] uint8
        if not self.inp_standardized:
            pred_imgs = (standardize(pred_imgs) * 255).type(torch.uint8)
            gt_imgs = (standardize(gt_imgs) * 255).type(torch.uint8)

        ### grayscale inputs -> expand channel dim for the Inception model
        if pred_imgs.shape[1] == 1:
            pred_imgs = pred_imgs.repeat(1, 3, 1, 1)
        if gt_imgs.shape[1] == 1:
            gt_imgs = gt_imgs.repeat(1, 3, 1, 1)

        self.fid.reset()
        self.fid.update(gt_imgs, real=True)
        self.fid.update(pred_imgs, real=False)

        return self.fid.compute().item()


class PixCorr(torch.nn.Module):
    def __init__(self, reduction="mean"):
        assert reduction in ["mean", "sum", "none"], f"Invalid reduction: {reduction}"
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def batchwise_pearson_correlation(Z, B):
        assert Z.ndim == 2 and B.ndim == 2, "Input tensors must be 2D"
        assert Z.shape[0] == B.shape[0], "Input tensors must have the same batch size"

        ### calculate and subtract means
        Z_mean = torch.mean(Z, dim=1, keepdim=True)
        B_mean = torch.mean(B, dim=1, keepdim=True)
        Z_centered = Z - Z_mean
        B_centered = B - B_mean

        ### calculate pearson correlation coefficient
        num = Z_centered @ B_centered.T
        Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
        B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
        denom = Z_centered_norm @ B_centered_norm.T

        pearson_corr = (num / (denom + 1e-6))

        return pearson_corr

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        assert len(preds) == len(targets), "Number of predicted images must match number of original images"

        targets_flat = targets.view(len(targets), -1)
        preds_flat = preds.view(len(preds), -1)

        corr_mean = self.batchwise_pearson_correlation(targets_flat, preds_flat).diag()

        if self.reduction == "mean":
            corr_mean = corr_mean.mean()
        elif self.reduction == "sum":
            corr_mean = corr_mean.sum()
        elif self.reduction == "none":
            pass

        return corr_mean


class TwoWayAlexNet(torch.nn.Module):
    """
    Modified from https://github.com/MedARC-AI/MindEyeV2

    Citation:
      Scotti, Tripathy, Torrico, Kneeland, Chen, Narang, Santhirasegaran, Xu, Naselaris, Norman, & Abraham.
      MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. International Conference on
      Machine Learning. (2024). arXiv:2403.11207
    """
    def __init__(
        self,
        inp_zscored: bool = False,
        feature_layers=["features.4", "features.11"],
        avg_across_layers: bool = False,
        reduction="mean",
        device: str = "cuda",
    ):
        assert reduction in ["mean", "sum", "none"], f"Invalid reduction: {reduction}"
        super().__init__()

        self.device = device
        self.inp_zscored = inp_zscored
        self.feature_layers = feature_layers
        self.avg_across_layers = avg_across_layers
        self.reduction = reduction

        ### feature extractor
        self.alex_model = create_feature_extractor(
            alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
            return_nodes=self.feature_layers,
        ).to(self.device)
        self.alex_model.eval().requires_grad_(False)

        ### preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if not self.inp_zscored:
            self.preprocess.transforms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

    @torch.no_grad()
    def two_way_identification(self, all_recons, all_images, feature_layer=None, return_avg=True):
        ### prepare data
        preds = self.alex_model(self.preprocess(all_recons).to(self.device))
        reals = self.alex_model(self.preprocess(all_images).to(self.device))
        if feature_layer is None:
            preds = preds.float().flatten(1).cpu().numpy()
            reals = reals.float().flatten(1).cpu().numpy()
        else:
            preds = preds[feature_layer].float().flatten(1).cpu().numpy()
            reals = reals[feature_layer].float().flatten(1).cpu().numpy()

        ### calculate correlations and success rates
        r = np.corrcoef(reals, preds)
        r = r[:len(all_images), len(all_images):]
        congruents = np.diag(r)

        success = r < congruents
        success_cnt = np.sum(success, 0)

        if return_avg:
            perf = np.mean(success_cnt) / (len(all_images) - 1)
            return perf
        else:
            return (success_cnt / (len(all_images) - 1))

    def forward(self, preds, targets):
        results = dict()
        for feature_layer in self.feature_layers:
            results[feature_layer] = self.two_way_identification(
                all_recons=preds,
                all_images=targets,
                feature_layer=feature_layer,
                return_avg=self.reduction == "mean",
            )
        if self.reduction == "mean":
            results = {k: v.mean() for k, v in results.items()}
        elif self.reduction == "sum":
            results = {k: v.sum() for k, v in results.items()}

        if self.avg_across_layers:
            return sum(results.values()) / len(results)

        return results


class BrainDistance(torch.nn.Module):
    def __init__(
        self,
        encoder,
        use_gt_resp: bool = True,
        resp_loss_fn = F.mse_loss,
        zscore_inp: bool = False,
        minmax_normalize_inp: bool = False,
        device="cuda",
    ):
        super().__init__()
        self.use_gt_resp = use_gt_resp
        self.zscore_inp = zscore_inp
        self.minmax_normalize_inp = minmax_normalize_inp
        self.resp_loss_fn = resp_loss_fn
        self.device = device
        self.encoder = encoder
        self.encoder.to(self.device)
        self.encoder.requires_grad_(False)

    def _compute_brain_similarity(self, stim_pred: Tensor, stim: Tensor, resp: Tensor, **kwargs) -> Tensor:
        resp = resp.to(self.device) if self.use_gt_resp else self.encoder(stim.to(self.device), **kwargs)
        stim_pred_resp = self.encoder(stim_pred.to(self.device), **kwargs)
        return self.resp_loss_fn(stim_pred_resp, resp)

    def forward(self, pred: Tensor, target: Tensor, resp: Tensor, **kwargs) -> Tensor:
        if self.zscore_inp:
            pred = normalize(pred)
            target = normalize(target)

        if self.minmax_normalize_inp:
            pred = standardize(pred)
            target = standardize(target)

        return self._compute_brain_similarity(stim_pred=pred, stim=target, resp=resp, **kwargs)

import torch
from torch import einsum
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.vgg import VGG19_Weights
from typing import Any
import numpy as np


class Lossfun:
    def __init__(self, alpha: float, beta_vgg: float, beta_pix: float, normalized: bool, device='cuda') -> None:
        self._alpha = alpha
        self._bce = nn.BCELoss()
        self._beta_vgg = beta_vgg
        self._beta_pix = beta_pix
        self._l1 = nn.L1Loss()
        self._vgg = VggLoss_Summed(normalized=normalized, device=device)

    def __call__(self, p: float, p_hat: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        if p_hat.shape[-1] == 1:
            p_hat = p_hat.squeeze(-1)
        dis_loss = self._alpha * torch.mean(self._bce(p_hat, torch.full_like(p_hat, p)))
        gen_loss_vgg = self._beta_vgg * torch.mean(self._vgg(y_hat, y))
        gen_loss_pix = self._beta_pix * torch.mean(self._l1(y_hat, y))
        total_loss = dis_loss + gen_loss_vgg + gen_loss_pix
        return total_loss, dis_loss, gen_loss_vgg, gen_loss_pix

class VggLoss_Summed(nn.Module):
    def __init__(self, normalized, device='cuda') -> None:
        super(VggLoss_Summed, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg19.features.to(device).eval()
        self.transformer = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        self.normalized = normalized
        self.layers = [2, 7, 16, 25, 34]
        self.warnings_raised = dict()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, reduction: str="mean") -> torch.Tensor:
        if not self.normalized and (y_hat.min() < 0 or y_hat.max() > 1 or y.min() < 0 or y.max() > 1):
            if not self.warnings_raised.get("y_hat", False):
                print("*****\n[WARNING]: y_hat and y should be in the range [0, 1]\n*****")
                self.warnings_raised["y_hat"] = True

        if y_hat.size(1) == 1:
            y_hat = y_hat.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)

        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        y_hat = F.interpolate(y_hat, size=(224, 224), mode='bilinear', align_corners=False)

        if not self.normalized:
            y = self.transformer(y)
            y_hat = self.transformer(y_hat)

        losses = []
        for i, layer in enumerate(self.layers):
            y_feat = self.features[:layer](y)
            y_hat_feat = self.features[:layer](y_hat)
            losses.append(F.mse_loss(y_feat, y_hat_feat, reduction=reduction))

        return sum(losses) / len(losses) if reduction != "none" else losses


class Identity(nn.Module):
    def __init__(self, count: int, depth: int) -> None:
        super(Identity, self).__init__()
        self.count = count
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Skip(nn.Module):
    def __init__(self, count: int, depth: int, layer: nn.Module, pool: bool) -> None:
        super(Skip, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(depth, layer.depth, 4, 2 if pool else 1, 1, bias=False),
            nn.BatchNorm2d(layer.depth, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True),
            layer,
            nn.ConvTranspose2d(layer.count, count, 4, 2 if pool else 1, 1, bias=False),
            nn.BatchNorm2d(count, momentum=0.1)
        )
        self._count = count
        self._depth = depth

    @property
    def count(self) -> int:
        return self._count + self._depth

    @property
    def depth(self) -> int:
        return self._depth

    def padded_cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Pad the smaller tensor to the size of the larger tensor and concatenate them """

        # Get shapes
        h1, w1 = x.shape[-2], x.shape[-1]
        h2, w2 = y.shape[-2], y.shape[-1]

        # Compute padding (assumes even padding on both sides)
        pad_h = max(h1, h2) - min(h1, h2)
        pad_w = max(w1, w2) - min(w1, w2)

        # Determine which tensor to pad
        if h1 < h2 or w1 < w2:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        else:
            y = F.pad(y, (0, pad_w, 0, pad_h))

        # Concatenate along channel dimension
        return torch.cat([x, y], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.padded_cat(x, self.block(x)))


class Network(nn.Module):
    def __init__(self, count: int, depth: int, normalized: bool) -> None:
        super(Network, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(depth, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer = Identity(512, 512)
        layer = Identity(256, 256)
        # layer = Skip(512, 512, layer, pool=False)
        layer = Skip(256, 256, layer, pool=False)
        layer = Skip(128, 128, layer, pool=True)
        layer = Skip(64, 64, layer, pool=True)
        self.main.add_module('layer', layer)
        self.main.add_module('final', nn.Sequential(
            nn.ConvTranspose2d(128, count, 4, 2, 1),
            nn.Identity() if normalized else nn.Sigmoid()
        ))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class InverseRetinotopicMapping(nn.Module):
    """ Learned inverse retinotopic mapping """

    def __init__(
        self,
        n_neurons: int,
        height: int = 36,
        width: int = 64,
        sum_maps: bool = True,
        device: str = "cuda",
    ) -> None:
        super(InverseRetinotopicMapping, self).__init__()
        self.width = width
        self.height = height
        self.sum_maps = sum_maps
        self.coords = nn.Parameter(torch.stack([
            torch.rand(n_neurons, 1, 1, device=device).mul(2).sub(1),  # x-coordinates (range: [-1, 1])
            torch.rand(n_neurons, 1, 1, device=device).mul(2).sub(1)  # y-coordinates (range: [-1, 1])
        ], dim=-1))  # Shape: (n_neurons, 1, 1, 2)
        self.sigma = nn.Parameter(torch.ones(n_neurons, 1, 1, 1, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### prepare grid
        y_cord = torch.arange(self.height, device=x.device, dtype=torch.float32).view(self.height, 1).expand(-1, self.width).div(self.height).mul(2).sub(1)
        x_cord = torch.arange(self.width, device=x.device, dtype=torch.float32).view(1, self.width).expand(self.height, -1).div(self.width).mul(2).sub(1)
        xy_grid = torch.stack([x_cord, y_cord], dim=-1).float().unsqueeze(0)

        # retinotopic_maps = (1. / (2. * 3.14159265359 * self.sigma)) * torch.exp(
        #     -torch.sum((xy_grid - self.coords)**2., dim=-1).unsqueeze(-1)
        #     / (2 * self.sigma)
        # ) # (n_neurons, height, width, 1)
        retinotopic_maps = torch.exp(
            -(xy_grid - self.coords).pow(2).div(2 * self.sigma.pow(2)).sum(dim=-1, keepdim=True)
        ) # (n_neurons, height, width, 1)

        ### min-max normalization
        # retinotopic_maps = retinotopic_maps - retinotopic_maps.amin(dim=(1, 2, 3), keepdim=True)
        # retinotopic_maps = retinotopic_maps / retinotopic_maps.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)

        ### contextualize retinotopic maps with input
        ### element-wise multiplication -> (batch, n_neurons, height, width)
        retinotopic_maps = \
            x.unsqueeze(-1).unsqueeze(-1) * retinotopic_maps.permute(3, 0, 1, 2)
        if self.sum_maps:
            retinotopic_maps = retinotopic_maps.sum(dim=1, keepdim=True)

        return retinotopic_maps


class Generator(nn.Module):
    def __init__(
        self,
        input_channels,
        normalized: bool,
        inverse_retinotopic_mapping_cfg: dict,
        alpha: float,
        beta_vgg: float,
        beta_pix: float,
        lr: float = 0.0002,
        betas: tuple = (0.5, 0.999),
        weight_decay: float = 0.0,
        device='cuda',
    ) -> None:
        super(Generator, self).__init__()
        self._lossfun = Lossfun(alpha=alpha, beta_vgg=beta_vgg, beta_pix=beta_pix, normalized=normalized, device=device)
        self._network = Network(count=1, depth=input_channels, normalized=normalized).to(device)
        if inverse_retinotopic_mapping_cfg is not None:
            self._inv_retinotopic_map = InverseRetinotopicMapping(**inverse_retinotopic_mapping_cfg)
        else:
            self._inv_retinotopic_map = nn.Identity()

        ### init optimizer
        inv_ret_map_coords = [p for n, p in self._inv_retinotopic_map.named_parameters() if "coords" in n]
        inv_ret_map_other = [p for n, p in self._inv_retinotopic_map.named_parameters() if "coords" not in n]
        self._trainer = optim.Adam([
            {"params": self._network.parameters()},
            {"params": inv_ret_map_coords, "weight_decay": 0.0, "lr": lr}, # do not apply weight decay to coords
            {"params": inv_ret_map_other, "weight_decay": 0.0, "lr": lr},
            ], lr=lr, betas=betas, weight_decay=weight_decay,
        )

        self.update_step = 0

    def step(self):
        self._trainer.step()
        self.update_step += 1
        self._trainer.zero_grad()

    def forward(self, x: torch.Tensor, return_inv_ret_maps: bool = False) -> torch.Tensor:
        inv_ret_maps = self._inv_retinotopic_map(x)
        return self._network(inv_ret_maps) if not return_inv_ret_maps else (self._network(inv_ret_maps), inv_ret_maps)

    def train_model(self, d: nn.Module, x: torch.Tensor, y: torch.Tensor, step: bool) -> float:
        self.train()

        ### forward pass
        y_hat, x_ret_maps = self(x, return_inv_ret_maps=True)
        p_hat = d(y_hat)
        total_loss, dis_loss, gen_loss_vgg, gen_loss_pix = self._lossfun(1, p_hat, y, y_hat)

        ### update
        self._trainer.zero_grad()
        total_loss.backward()
        if step:
            self.step()

        return float(total_loss.item()), float(dis_loss.item()), float(gen_loss_vgg.item()), float(gen_loss_pix.item())


if __name__ == "__main__":
    inv_map = InverseRetinotopicMapping(n_neurons=100, height=36, width=64, sum_maps=True, device="cuda")
    x = torch.randn(4, 100).to("cuda")
    y = inv_map(x)
    print(y.shape)
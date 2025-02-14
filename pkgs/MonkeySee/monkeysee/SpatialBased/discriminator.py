import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List


class History:
    """ A class to maintain a history of tensors with a fixed capacity. It allows for random replacement
    of elements when the capacity is reached.

    Attributes:
        _capacity (int): The maximum number of elements to store in the history.
        _data (List[torch.Tensor]): The list to store the tensors.

    Methods:
        __init__(capacity: int) -> None:
            Initializes the History object with a given capacity.
        
        __call__(z_prime: torch.Tensor) -> torch.Tensor:
            Adds elements from the input tensor to the history, replacing existing elements
            randomly if the capacity is reached, and returns a tensor of selected elements.
        
        capacity() -> int:
            Returns the capacity of the history.
        
        data() -> List[torch.Tensor]:
            Returns the list of stored tensors.
    """ 
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data = []

    def __call__(self, z_prime: torch.Tensor) -> torch.Tensor:
        z = []
        for i in range(z_prime.size(0)):
            if len(self._data) < self._capacity:
                z.append(z_prime[i])
                self._data.append(z_prime[i])
            elif torch.rand(1).item() < 0.5:
                idx = int(torch.randint(0, len(self._data), (1,)).item())
                z.append(self._data.pop(idx))
                self._data.append(z_prime[i])
            else:
                z.append(z_prime[i])
        return torch.stack(z, dim=0)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def data(self) -> List[torch.Tensor]:
        return self._data


class Lossfun:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._bce = nn.BCELoss()

    def __call__(self, p: float, p_hat: torch.Tensor) -> torch.Tensor:
        target = torch.full_like(p_hat, p, device=p_hat.device)
        return self._alpha * self._bce(p_hat.clip(0, 1), target)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def bce(self) -> nn.BCELoss:
        return self._bce


class Network(nn.Module):
    def __init__(self, count: int, depth: int) -> None:
        super(Network, self).__init__()
        self._count = count
        self._depth = depth

        self.layers = nn.Sequential(
            nn.Conv2d(depth, 64, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, count, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(28, 1),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    @property
    def count(self) -> int:
        return self._count

    @property
    def depth(self) -> int:
        return self._depth


class Discriminator(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lr: float = 0.0002,
        betas: tuple = (0.5, 0.999),
        weight_decay: float = 0.0,
        device: str = 'cuda',
    ) -> None:
        super(Discriminator, self).__init__()
        self._history = History(50)
        self._device = device
        self._lossfun = Lossfun(1.0)
        self._network = Network(1, input_channels).to(self.device)
        self._optimizer = optim.Adam(self._network.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    @property
    def history(self) -> History:
        return self._history

    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> Network:
        return self._network

    @property
    def optimizer(self) -> optim.Adam:
        return self._optimizer

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)

    def step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

    def train_model(self, g: nn.Module, x: torch.Tensor, y: torch.Tensor, step: bool) -> float:
        """ Train the discriminator using the generator `g`, real input `x`, and real output `y`. """
        self.train()

        g, x, y = g.to(self.device), x.to(self.device), y.to(self.device)

        ### generate fake data
        fake_y = g(x).detach()

        ### compute loss for real and fake samples
        fake_ys = self._history(fake_y)
        real_loss = self._lossfun(1, self(y))
        fake_loss = self._lossfun(0, self(fake_ys))
        total_loss = 0.5 * (real_loss + fake_loss)

        ### update
        self._optimizer.zero_grad()
        total_loss.backward()
        if step:
            self.step()

        return total_loss.item()

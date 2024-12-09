import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        # Linear regression layer
        self.linear = nn.Sequential(
            nn.Linear(9395, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 4 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (4, 16, 16)),
            nn.Dropout(),
        )

    def forward(self, x):
        return self.linear(x)

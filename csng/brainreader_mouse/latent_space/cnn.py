import torch
import torch.nn as nn


class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        # Linear regression layer
        self.linear = nn.Sequential(
            nn.Linear(9395, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 4 * 32 * 32),
            nn.Unflatten(1, (4, 32, 32)),
            nn.Dropout(),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                4, 8, kernel_size=3, stride=1, padding=1
            ),  # Increase channels from 4 to 8
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(
                8, 16, kernel_size=3, stride=1, padding=1
            ),  # Increase channels from 8 to 16
            nn.ReLU(),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=1, padding=1
            ),  # Increase channels from 16 to 32
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Increase channels from 32 to 64
            nn.ReLU(),
            nn.Conv2d(
                64, 4, kernel_size=3, stride=1, padding=1
            ),  # Reduce channels to 1 for the final output (if needed)
        )

    def forward(self, x):
        return self.linear(x)

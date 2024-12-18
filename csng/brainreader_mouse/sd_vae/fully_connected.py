import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size=9395):
        super(FullyConnected, self).__init__()
        # Linear regression layer
        self.linear = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.Unflatten(1, (4, 8, 8)),
        )

    def forward(self, x):
        return self.linear(x)

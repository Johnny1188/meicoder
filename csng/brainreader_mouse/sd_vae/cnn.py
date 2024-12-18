import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_size=9395):
        super(CNN, self).__init__()
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
            nn.Unflatten(1, (16, 8, 8)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )

        self.relu = nn.ReLU()

        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.linear(x)
        for _ in range(2):
            x_new = self.conv(x)
            x = self.relu(x + x_new)

        return self.output_layer(x)

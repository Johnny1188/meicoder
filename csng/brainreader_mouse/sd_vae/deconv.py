import torch.nn as nn


class Deconv(nn.Module):
    def __init__(self, input_size=9395, output_size=3, dropout=0.5):
        super(Deconv, self).__init__()
        self.input_size = input_size

        # Linear regression layer
        self.linear = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(1, 16),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 64),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_layer = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.reshape(-1, self.input_size, 1)
        x = self.linear(x)

        x = x.reshape(-1, self.input_size, 8, 8)
        x = self.conv_layers(x)

        return self.output_layer(x)

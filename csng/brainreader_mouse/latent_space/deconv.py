import torch.nn as nn


class Deconv(nn.Module):
    def __init__(self, activation="relu", output_size=3):
        super(Deconv, self).__init__()
        # Linear regression layer
        self.linear = nn.Linear(1, 16)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(9395, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        activation_layer = (
            nn.Sigmoid() if activation == "sigmoid" else nn.ReLU()
        )
        print(
            f"Using {activation_layer} as the activation layer for convolutional network."
        )
        deconv_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(4),
                activation_layer,
                nn.Dropout(0.3),
            )
            for _ in range(output_size - 2)
        ]
        deconv_layers.append(
            nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1)
        )

        self.enlarge = nn.Sequential(*deconv_layers)

    def forward(self, x):
        x = x.reshape(-1, 9395, 1)
        x = self.linear(x)

        x = x.reshape(-1, 9395, 4, 4)
        x = self.conv_layers(x)

        x = self.enlarge(x)

        return x

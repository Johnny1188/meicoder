import torch.nn as nn


class DeconvSeparate(nn.Module):
    def __init__(self, output_size=3):
        super(DeconvSeparate, self).__init__()
        # Linear regression layer
        self.linear = nn.Linear(1, 16)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(9395, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
        )

        deconv_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    9395, 9395, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            for _ in range(output_size - 2)
        ]
        deconv_layers.append(
            nn.ConvTranspose2d(9395, 9395, kernel_size=4, stride=2, padding=1)
        )

        self.enlarge = nn.Sequential(*deconv_layers)

    def forward(self, x):
        x = x.reshape(-1, 9395, 1)
        x = self.linear(x)

        x = x.reshape(-1, 9395, 4, 4)
        x = self.enlarge(x)

        x = self.conv_layers(x)

        return x

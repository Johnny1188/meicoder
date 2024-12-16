import torch
from torch import nn
from torchvision import transforms
from math import prod


import os

device = os.environ["DEVICE"]
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils', )

class ResnetExtractor:
    def __init__(self):
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.eval().to(device)
        self.resnet_layer = self.resnet50.layers[2][0].downsample[0]

    def get_features(self, img, verbose = False):
        IMGNET_NORM = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # imgnet_input = F.rgb_to_grayscale(IMGNET_NORM(img), num_output_channels=3).to(device).nsqueeze(0)

        def assign_features(module, input, output):
            nonlocal features
            features = output
        hook = self.resnet_layer.register_forward_hook(assign_features)
        
        with torch.no_grad():
            raw_output = self.resnet50(img)
            if verbose:
                output = torch.nn.functional.softmax(raw_output, dim=1)
                results = utils.pick_n_best(predictions=output, n=5)
                for i, result in enumerate(results):
                    show_image(img[i], result)



        hook.remove()
        assert features is not None
        features = features.squeeze(0)
        return features

class ReadIn(torch.nn.Module):
    def __init__(self, n_features, target_shape, alpha=1.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.alpha = alpha
        self.target_shape = target_shape
        self.inter_shape = (128, 14, 14)

        self.model = torch.nn.Sequential(
            # Linear layer to map input to intermediate feature space
            torch.nn.Linear(n_features, prod(self.inter_shape), bias=True, device=device, dtype=dtype),
            torch.nn.LeakyReLU(0.1, inplace=True),

            # Reshape to intermediate shape
            torch.nn.Unflatten(1, self.inter_shape),

            # Conv1: 128 -> 256 with BatchNorm and ReLU
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),

            # Conv2: 256 -> 512 with BatchNorm and ReLU
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias=True),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1024),
            # torch.nn.LeakyReLU(0.1, inplace=True),
        )
        self.apply(initialize_weights)
        
    def forward(self, x):
        return self.model(x)

def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        # Initialize weights close to zero
        torch.nn.init.uniform_(module.weight, a=-0.000001, b=0.000001)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, torch.nn.Conv2d):
        # Initialize weights close to zero
        torch.nn.init.uniform_(module.weight, a=-0.000001, b=0.000001)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, torch.nn.BatchNorm2d):
        # BatchNorm weight and bias initialization
        torch.nn.init.constant_(module.weight, 1.0)
        torch.nn.init.constant_(module.bias, 0.0)

class UpsampleModel(nn.Module):
    def __init__(self):
        super(UpsampleModel, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=768,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                in_channels=768, out_channels=512,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=384,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                in_channels=384, out_channels=256,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)
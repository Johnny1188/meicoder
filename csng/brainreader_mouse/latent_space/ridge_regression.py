import torch
import torch.nn as nn


class RidgeRegression(nn.Module):
    def __init__(self, lambda_reg=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(9395, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4 * 64 * 64),
            nn.Unflatten(1, (4, 64, 64)),
        )
        self.lambda_reg = lambda_reg

    def forward(self, x):
        preds = self.linear(x)
        return preds.reshape(-1, *self.output_dim)

    def get_regularization_loss(self):
        """
        Calculate the L2 regularization loss (Ridge term) as lambda * ||w||^2
        This is the sum of squares of the weights (excluding bias term).
        """
        # Get the weights from the model, excluding the bias term
        weights = self.linear.weight
        regularization_loss = self.lambda_reg * torch.sum(weights**2)
        return regularization_loss

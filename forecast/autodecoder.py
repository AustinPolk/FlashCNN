import torch
from torch import nn, optim
import numpy as np

class AutoDecoder(nn.Module):
    def __init__(self, num_features, num_latent_features):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, num_latent_features)
        )
        self.encoder = nn.Sequential(
            nn.Linear(num_latent_features, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, num_features)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        encoded = self.encoder(decoded)
        return encoded
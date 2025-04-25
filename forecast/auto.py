import torch
from torch import nn, optim
import numpy as np

class Decoder(nn.Module):
    def __init__(self, num_features, num_latent_features):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_latent_features)
        )
        self.encoder = nn.Sequential(
            nn.Linear(num_latent_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_features)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        encoded = self.encoder(decoded)
        return encoded
    
class Encoder(nn.Module):
    def __init__(self, num_features, num_latent_features):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_latent_features)
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_latent_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
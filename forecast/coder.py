import torch
from torch import nn, optim
import numpy as np

class Coder(nn.Module):
    def __init__(self, num_features, num_latent_features, t_size=12):
        super(Coder, self).__init__()
        self.incoder = nn.Sequential(
            nn.Linear(num_features, t_size),
            nn.ReLU(),
            nn.Linear(t_size, num_latent_features)
        )
        self.outcoder = nn.Sequential(
            nn.Linear(num_latent_features, t_size),
            nn.ReLU(),
            nn.Linear(t_size, num_features)
        )

    def forward(self, x):
        incoded = self.incoder(x)
        outcoded = self.outcoder(incoded)
        return outcoded
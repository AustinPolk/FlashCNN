import torch
from torch import nn, optim
import numpy as np

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_features, seq_len):
        self.modules = nn.Sequential()

        # setup the actual thing
        self.modules.add_module('Conv1', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.modules.add_module('ReLU1', nn.ReLU())
        self.modules.add_module('MaxPool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.modules.add_module('Conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.modules.add_module('ReLU2', nn.ReLU())
        self.modules.add_module('MaxPool2', nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.modules.add_module('Conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.modules.add_module('ReLU3', nn.ReLU())
        self.modules.add_module('MaxPool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.modules.add_module('Flatten', nn.Flatten())
        self.modules.add_module('FC1', nn.Linear(50176, 256))
        self.modules.add_module('ReLU10', nn.ReLU())
        self.modules.add_module('FC2', nn.Linear(256, 256))
        self.modules.add_module('ReLU11', nn.ReLU())

    def forward(self, x):
        pass
import torch
from torch import nn, optim
import numpy as np

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_features, sequence_len, consecutive_days=3):
        super(TimeSeriesCNN, self).__init__()
        self.network = nn.Sequential()

        # first convolutional layer, followed by relu and maxpooling layers
        self.network.add_module('Conv1', nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, num_features)))
        self.network.add_module('Conv2', nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(consecutive_days, 1)))
        self.network.add_module('ReLU1', nn.ReLU())
        self.network.add_module('MaxPool1', nn.MaxPool2d(kernel_size=(2,1), stride=2))

        # next layer, same thing.
        self.network.add_module('Conv3', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(consecutive_days, 1)))
        self.network.add_module('ReLU2', nn.ReLU())
        self.network.add_module('MaxPool2', nn.MaxPool2d(kernel_size=(2,1), stride=2))

        # then flatten everything into a single dimension and apply dropout
        self.network.add_module('Flatten', nn.Flatten())
        self.network.add_module('Dropout', nn.Dropout(0.2))

        # fully connected layers to produce a new set of features
        flat_size = 16 * ((sequence_len - consecutive_days + 1) // 2 - consecutive_days + 1) // 2
        self.network.add_module('FC1', nn.Linear(flat_size, 128))
        self.network.add_module('ReLU3', nn.ReLU())
        self.network.add_module('FC2', nn.Linear(128, num_features))

    def forward(self, x):
        return self.network(x)
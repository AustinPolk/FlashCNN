import torch
from torch import nn
import numpy as np

class FlashModel(nn.Module):
    def __init__(self, input_shape, output_shape, stations, k=5, dropout=0.2):
        super(FlashModel, self).__init__()
        
        self.channels = input_shape[0]
        self.sequence_length = input_shape[1]
        self.input_features = input_shape[2]
        self.output_features = output_shape[0]
        self.kernel_size = k
        self.dropout_rate = dropout
        self.stations = stations
        self.network = self._create_network()

    def _create_network(self):
        network = nn.Sequential()
    
        c, h, w = self.channels, self.sequence_length, self.input_features
        print(c, h, w)
        
        # first convolutional layer, followed by relu and maxpooling layers
        network.add_module('Conv1', nn.Conv2d(in_channels=self.channels, out_channels=4*self.channels, kernel_size=(1, self.input_features)))
        c, h, w = 4*c, h, 1
        print(c, h, w)
    
        network.add_module('Conv2', nn.Conv2d(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=(self.kernel_size, 1)))
        c, h = 2*c, h - self.kernel_size + 1
        print(c, h, w)
        
        network.add_module('ReLU1', nn.ReLU())
        network.add_module('MaxPool1', nn.MaxPool2d(kernel_size=(2,1), stride=2))
        h = h / 2
        print(c, h, w)
        
        # next layer, same thing.
        network.add_module('Conv3', nn.Conv2d(in_channels=8*self.channels, out_channels=16*self.channels, kernel_size=(self.kernel_size, 1)))
        c, h = 2*c, h - self.kernel_size + 1
        print(c, h, w)
        
        network.add_module('ReLU2', nn.ReLU())
        network.add_module('MaxPool2', nn.MaxPool2d(kernel_size=(2,1), stride=2))
        h = h / 2
        print(c, h, w)
        
        
        # then flatten everything into a single dimension and apply dropout
        network.add_module('Flatten', nn.Flatten())
        print(c * h * w)
        
        network.add_module('Dropout', nn.Dropout(self.dropout_rate))
        
        # fully connected layers to produce a new set of features
        network.add_module('FC1', nn.Linear(int(c*h*w), 512))
        network.add_module('ReLU3', nn.ReLU())
        network.add_module('FC2', nn.Linear(512, 128))
        network.add_module('ReLU4', nn.ReLU())
        network.add_module('FC3', nn.Linear(128, self.output_features))
    
        return network
    def forward(self, x):
        return self.network(x)
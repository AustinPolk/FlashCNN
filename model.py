import torch
from torch import nn
import numpy as np

class FlashModel(nn.Module):
    def __init__(self, input_shape, output_shape, stations, k=5, pool_k=2, dropout=0.2, convolutional_layers=2, channel_multiplier=2, variables_per_station=6):
        super(FlashModel, self).__init__()
        
        self.channels = input_shape[0]
        self.sequence_length = input_shape[1]
        self.input_features = input_shape[2]
        self.output_features = output_shape[0]
        self.kernel_size = k
        self.pool_kernel_size = pool_k
        self.dropout_rate = dropout
        self.stations = stations
        self.convolutional_layers = convolutional_layers
        self.channel_multiplier = channel_multiplier
        self.variables_per_station = variables_per_station
        self.network = self._create_network()
    
    def _create_network(self):
        network = nn.Sequential()
    
        c, h, w = self.channels, self.sequence_length, self.input_features
        print(c, h, w)
        
        # first convolutional layer, followed by relu and maxpooling layers
        network.add_module('Conv1', nn.Conv2d(in_channels=self.channels, out_channels=2*self.channel_multiplier*c, kernel_size=(1, self.input_features)))
        c, h, w = 2*self.channel_multiplier*c, h, 1
        print(c, h, w)

        # additional convolutional layers + ReLU + pooling as demanded by the user
        for i in range(self.convolutional_layers):
            network.add_module(f'Conv{i + 2}', nn.Conv2d(in_channels=c, out_channels=self.channel_multiplier*c, kernel_size=(self.kernel_size, 1)))
            c, h = self.channel_multiplier*c, h - self.kernel_size + 1
            print(c, h, w)
            
            network.add_module(f'ReLU{i+1}', nn.ReLU())
            network.add_module(f'MaxPool{i+1}', nn.MaxPool2d(kernel_size=(self.pool_kernel_size, 1), stride=self.pool_kernel_size))
            h = h // self.pool_kernel_size
            print(c, h, w)
        
        # then flatten everything into a single dimension and apply dropout
        network.add_module('Flatten', nn.Flatten())
        print(c * h * w)
        
        network.add_module('Dropout', nn.Dropout(self.dropout_rate))
        
        # fully connected layers to produce a new set of features
        network.add_module('FC1', nn.Linear(int(c*h*w), 512 + self.output_features))
        network.add_module('ReLUFC1', nn.ReLU())
        network.add_module('FC2', nn.Linear(512 + self.output_features, 128 + self.output_features))
        network.add_module('ReLUFC2', nn.ReLU())
        network.add_module('FC3', nn.Linear(128 + self.output_features, self.output_features))
    
        return network
    def forward(self, x):
        y = self.network(x)
        return y
                
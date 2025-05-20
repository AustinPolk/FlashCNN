import torch
from torch import nn
import numpy as np

class FlashModel(nn.Module):
    def __init__(self, stations, variables_per_station, lookback, lookahead):
        super(FlashModel, self).__init__()
        
        self.channels = 2*stations
        self.sequence_length = lookback
        self.input_features = 2*variables_per_station + 3
        self.output_features = variables_per_station
        self.output_length = lookahead
        self.convolutional = self._create_convolutional()
        self.fully_connected = [self._create_fully_connected() for x in range(stations)]
    
    def _create_convolutional(self):
        network = nn.Sequential()
    
        c, cc, h, w = self.channels, 4*self.channels, self.sequence_length, self.input_features
        
        # compress features along the width of the input matrix using a conv kernel
        network.add_module('Conv1', nn.Conv2d(in_channels=self.channels, out_channels=cc, kernel_size=(1, w)))
        c, cc, h, w = cc, 2*cc, h, 1

        # perform convolutions along the remaining dimension, use ReLU and maxpooling
        network.add_module(f'Conv2', nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(5, 1)))
        network.add_module(f'ReLU2', nn.ReLU())
        network.add_module(f'MaxPool2', nn.MaxPool2d(kernel_size=(2, 1), stride=2))
        c, cc, h = cc, 2*cc, (h - 4)//2

        network.add_module(f'Conv3', nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(5, 1)))
        network.add_module(f'ReLU3', nn.ReLU())
        network.add_module(f'MaxPool3', nn.MaxPool2d(kernel_size=(2, 1), stride=2))
        c, cc, h = cc, 2*cc, (h - 4)//2

        # flatten output and apply dropout
        network.add_module('Flatten', nn.Flatten())
        network.add_module('Dropout', nn.Dropout(0.2))

        # return the network and the output size
        return network, int(c*h)

    def _create_fully_connected(self, input_size):
        network = nn.Sequential()

        # apply fully connected layers to output of convolutional layers
        network.add_module('FC1', nn.Linear(input_size, 512))
        network.add_module('ReLUFC1', nn.ReLU())
        network.add_module('FC2', nn.Linear(512, 128))
        network.add_module('ReLUFC2', nn.ReLU())
        network.add_module('FC3', nn.Linear(128, self.output_features*self.output_length))
        
        # unflatten the output to stack the predictions for each day (unflatten dim=1 since it is expected to be batched in dim=0)
        network.add_module('Unflatten', nn.Unflatten(dim=1, unflattened_size=(self.output_length, self.output_features)))
    
        return network
    def forward(self, x):
        
        conv = self.convolutional(x)
        station_outputs = [fc(conv) for fc in self.fully_connected]
        y = torch.stack(station_outputs, dim=1) # stack in dim=1 since dim=0 is assumed as batch dimension

        return y
                
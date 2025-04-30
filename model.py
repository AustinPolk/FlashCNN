import torch
from torch import nn
import numpy as np

def create_model(num_features, sequence_len, channels, k):
    network = nn.Sequential()

    c, h, w = channels, sequence_len, num_features
    print(c, h, w)
    
    # first convolutional layer, followed by relu and maxpooling layers
    network.add_module('Conv1', nn.Conv2d(in_channels=channels, out_channels=4*channels, kernel_size=(1, num_features)))
    c, h, w = 4*c, h, 1
    print(c, h, w)

    network.add_module('Conv2', nn.Conv2d(in_channels=4*channels, out_channels=8*channels, kernel_size=(k, 1)))
    c, h = 2*c, h - k + 1
    print(c, h, w)
    
    network.add_module('ReLU1', nn.ReLU())
    network.add_module('MaxPool1', nn.MaxPool2d(kernel_size=(2,1), stride=2))
    h = h / 2
    print(c, h, w)
    
    # next layer, same thing.
    network.add_module('Conv3', nn.Conv2d(in_channels=8*channels, out_channels=16*channels, kernel_size=(k, 1)))
    c, h = 2*c, h - k + 1
    print(c, h, w)
    
    network.add_module('ReLU2', nn.ReLU())
    network.add_module('MaxPool2', nn.MaxPool2d(kernel_size=(2,1), stride=2))
    h = h / 2
    print(c, h, w)
    
    
    # then flatten everything into a single dimension and apply dropout
    network.add_module('Flatten', nn.Flatten())
    print(c * h * w)
    
    network.add_module('Dropout', nn.Dropout(0.2))
    
    # fully connected layers to produce a new set of features
    #flat_size = 16 * channels * ((sequence_len - k + 1) // 2 - k + 1) // 2
    network.add_module('FC1', nn.Linear(int(c*h*w), 512))
    network.add_module('ReLU3', nn.ReLU())
    network.add_module('FC2', nn.Linear(512, 128))
    network.add_module('ReLU4', nn.ReLU())
    network.add_module('FC3', nn.Linear(128, num_features))

    return network
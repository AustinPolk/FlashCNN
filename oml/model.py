from torch import nn

# a single FlashCNN module that models the weather at one specific "point" (station, grid cell, etc.)
class FlashPoint(nn.Module):
    def __init__(self, num_variables, lookback, lookahead, **parameters):
        super(FlashPoint, self).__init__()
        
        self.input_length = lookback
        self.input_features = num_variables
        self.output_features = num_variables
        self.output_length = lookahead
        self.params = parameters
        
        if 'k' not in self.params:
            self.params['k'] = 5
        if 'p' not in self.params:
            self.params['p'] = 2
        if 'z' not in self.params:
            self.params['z'] = 8
        if 'switch' not in self.params:
            self.params['switch'] = True
        if 'h' not in self.params:
            self.params['h'] = 256
        if 'l' not in self.params:
            self.params['l'] = 1
        
        self.convolutional, output_size = self._create_convolutional(self.params['k'], self.params['p'], self.params['z'], self.params['switch'], self.params['l'])
        self.fully_connected = self._create_fully_connected(output_size, self.params['h'])
    
    def _create_convolutional(self, k, p, z, switch, l):
        network = nn.Sequential()
    
        c, cc, h, w = 1, z, self.input_length, self.input_features
        
        if switch:
            # compress features along the width of the input matrix using a conv kernel
            network.add_module('Conv1', nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(1, w)))
            c, cc, h, w = cc, 2*cc, h, 1

            # perform convolutions along the remaining dimension, use ReLU and maxpooling
            network.add_module(f'Conv2', nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(k, 1)))
            network.add_module(f'ReLU2', nn.ReLU())
            network.add_module(f'MaxPool2', nn.MaxPool2d(kernel_size=(p, 1), stride=p))
            c, cc, h = cc, 2*cc, (h - k + 1)//p
        else:
            # optionally combine the above two kernels into a single, larger one
            network.add_module(f'Conv2', nn.Conv2d(in_channels=c, out_channels=2*cc, kernel_size=(k, w)))
            network.add_module(f'ReLU2', nn.ReLU())
            network.add_module(f'MaxPool2', nn.MaxPool2d(kernel_size=(p, 1), stride=p))
            c, cc, h, w = 2*cc, 4*cc, (h - k + 1)//p, 1

        for idx in range(l):
            network.add_module(f'Conv{3+idx}', nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(k, 1)))
            network.add_module(f'ReLU{3+idx}', nn.ReLU())
            network.add_module(f'MaxPool{3+idx}', nn.MaxPool2d(kernel_size=(p, 1), stride=p))
            c, cc, h = cc, 2*cc, (h - k + 1)//p

        # flatten output and apply dropout
        network.add_module('Flatten', nn.Flatten())
        network.add_module('Dropout', nn.Dropout(0.2))

        # return the network and the output size
        return network, int(c*h)

    def _create_fully_connected(self, input_size, h):
        network = nn.Sequential()

        # apply fully connected layers to output of convolutional layers
        network.add_module('FC1', nn.Linear(input_size, h))
        network.add_module('ReLUFC1', nn.ReLU())
        network.add_module('FC2', nn.Linear(h, h//4))
        network.add_module('ReLUFC2', nn.ReLU())
        network.add_module('FC3', nn.Linear(h//4, self.output_features*self.output_length))
        
        # unflatten the output to stack the predictions for each day (unflatten dim=1 since it is expected to be batched in dim=0)
        network.add_module('Unflatten', nn.Unflatten(dim=1, unflattened_size=(self.output_length, self.output_features)))
    
        return network
    def forward(self, x):
        # run the input through the convolutional layers, then through the fully connected layers
        conv = self.convolutional(x)
        y = self.fully_connected(conv)
        return y
    def copy(self):
        m = FlashPoint(self.input_features, self.input_length, self.output_length, **self.params)
        m.load_state_dict(self.state_dict())
        return m
    def copy_from(self, other):
        self.load_state_dict(other.state_dict())
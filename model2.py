import torch
from torch import nn
import numpy as np
import pandas as pd

# contains
class FlashNormals:
    def __init__(self, normals_dataframe: pd.DataFrame):
        self.normals = normals_dataframe # dataframe with daily normals, indexed by day of year, station, and normal type (mean, max, min)

    def mean(self, date: pd.Timestamp, station: str, variable: str):
        day_of_year = date.day_of_year
        day_of_year = day_of_year if day_of_year < 366 else 1
        return self.normals.loc[('mean', station, day_of_year), variable]

    def means(self, dates: list[pd.Timestamp], stations: list[str], variables: list[str]):
        out = torch.zeros((len(stations), len(dates), len(variables)))
        for i, station in enumerate(stations):
            for j, date in enumerate(dates):
                for k, variable in enumerate(variables):
                    out[i, j, k] = self.mean(date, station, variable)

    def max(self, date: pd.Timestamp, station: str, variable: str):
        day_of_year = date.day_of_year
        day_of_year = day_of_year if day_of_year < 366 else 1
        return self.normals.loc[('max', station, day_of_year), variable]

    def maxes(self, dates: list[pd.Timestamp], stations: list[str], variables: list[str]):
        out = torch.zeros((len(stations), len(dates), len(variables)))
        for i, station in enumerate(stations):
            for j, date in enumerate(dates):
                for k, variable in enumerate(variables):
                    out[i, j, k] = self.max(date, station, variable)

    def min(self, date: pd.Timestamp, station: str, variable: str):
        day_of_year = date.day_of_year
        day_of_year = day_of_year if day_of_year < 366 else 1
        return self.normals.loc[('min', station, day_of_year), variable]

    def mins(self, dates: list[pd.Timestamp], stations: list[str], variables: list[str]):
        out = torch.zeros((len(stations), len(dates), len(variables)))
        for i, station in enumerate(stations):
            for j, date in enumerate(dates):
                for k, variable in enumerate(variables):
                    out[i, j, k] = self.min(date, station, variable)

# wraps base model and reformats/clamps output so that it can be used as input to next iteration
class FlashModelWrapper:
    def __init__(self, stations: list[str], station_variables: list[str], look: tuple[int], normals: FlashNormals):
        self.model = FlashModel(len(stations), len(station_variables), look[0], look[1])
        self.stations = stations
        self.variables = station_variables
        self.normals = normals

    # need to get a tensor like (S, d, 3)
    def _dates_to_features(self, dates: list[pd.Timestamp]):
        days = torch.zeros(len(dates))
        for i, date in enumerate(dates):
            days[i] = date.day_of_year / 365.25
        sin = torch.sin(2*torch.pi*days)
        cos = torch.cos(2*torch.pi*days)
        times = torch.stack((days, sin, cos), dim=1)
        
        return times.unsqueeze(0).repeat(len(self.stations), 1)

    def __call__(self, x: torch.Tensor, dates_for_predicted_range: list[pd.Timestamp]):
        # assumes that x is unbatched
        model_output = self.model(x.unsqueeze(0)) #shape: (batch, stations, lookahead, features)
        model_output = model_output[-1] #remove batch dimension
        normals = self.normals.means(dates_for_predicted_range, self.stations, self.variables) #shape: (stations, lookahead, features)
        maxes = self.normals.maxes(dates_for_predicted_range, self.stations, self.variables) #shape: (stations, lookahead, features)
        mins = self.normals.mins(dates_for_predicted_range, self.stations, self.variables) #shape: (stations, lookahead, features)

        min_clamped = torch.maximum(model_output, mins)
        max_clamped = torch.minimum(min_clamped, maxes)

        difference = max_clamped - normals
        squared_difference = difference.square()

        first_channels = torch.concat((max_clamped, normals), dim=1)
        second_channels = torch.concat((difference, squared_difference), dim=1)

        time_features = self._dates_to_features(dates_for_predicted_range)

        first = torch.concat((first_channels, time_features), dim=2)
        second = torch.concat((second_channels, time_features), dim=2)
        total = torch.concat((first, second), dim=0)

        return torch.concat((x, total), dim=0)

# base model behind prediction
class FlashModel(nn.Module):
    def __init__(self, stations, variables_per_station, lookback, lookahead):
        super(FlashModel, self).__init__()
        
        self.channels = 2*stations
        self.input_length = lookback
        self.input_features = 2*variables_per_station + 3
        self.output_features = variables_per_station
        self.output_length = lookahead
        self.convolutional = self._create_convolutional()
        self.fully_connected = [self._create_fully_connected() for _ in range(stations)]
    
    def _create_convolutional(self):
        network = nn.Sequential()
    
        c, cc, h, w = self.channels, 4*self.channels, self.input_length, self.input_features
        
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
        
        # run the input through the convolutional layers, then through the independent fully connected layers
        conv = self.convolutional(x)
        station_outputs = [fc(conv) for fc in self.fully_connected]
        y = torch.stack(station_outputs, dim=1) # stack in dim=1 since dim=0 is assumed as batch dimension

        return y
                
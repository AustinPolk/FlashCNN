import torch
from torch import nn, optim
import numpy as np

class TimeSeriesLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_features)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
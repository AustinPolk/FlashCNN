from torch import nn
import numpy as np
from argparse import ArgumentParser
import json
import pickle

class FlashModelConfig:
    def __init__(self, **kwargs):
        self.model_name: str = kwargs['model_name'] if 'model_name' in kwargs else None 
        self.input_station_count: int = kwargs['input_station_count'] if 'input_station_count' in kwargs else None
        self.output_station_count: int = kwargs['output_station_count'] if 'output_station_count' in kwargs else None
        self.input_variables_per_station: int = kwargs['input_variables_per_station'] if 'input_variables_per_station' in kwargs else None
        self.output_variables_per_station: int = kwargs['output_variables_per_station'] if 'output_variables_per_station' in kwargs else None
        self.lookback_days: int = kwargs['lookback_days'] if 'lookback_days' in kwargs else None
        self.lookahead_days: int = kwargs['lookahead_days'] if 'lookahead_days' in kwargs else None
        self.convolutional_kernel_size: int = kwargs['convolutional_kernel_size'] if 'convolutional_kernel_size' in kwargs else None
        self.pooling_kernel_size: int = kwargs['pooling_kernel_size'] if 'pooling_kernel_size' in kwargs else None
        self.convolutional_kernel_activation: str = kwargs['convolutional_kernel_activation'] if 'convolutional_kernel_activation' in kwargs else None
        self.interlayer_channel_multiplier: int = kwargs['interlayer_channel_multiplier'] if 'interlayer_channel_multiplier' in kwargs else None
        self.convolutional_layer_count: int = kwargs['convolutional_layer_count'] if 'convolutional_layer_count' in kwargs else None
        self.dropout_rate: float = kwargs['dropout_rate'] if 'dropout_rate' in kwargs else None
        self.input_channel_count: int = kwargs['input_channel_count'] if 'input_channel_count' in kwargs else None
        self.fully_connected_activation: str = kwargs['fully_connected_activation'] if 'fully_connected_activation' in kwargs else None
        self.use_average_pooling: bool = kwargs['use_average_pooling'] if 'use_average_pooling' in kwargs else None
        self.fully_connected_layer_sizes: list = kwargs['fully_connected_layer_sizes'] if 'fully_connected_layer_sizes' in kwargs else None
        self.use_day_of_period_feature: bool = kwargs['use_day_of_period_feature'] if 'use_day_of_period_feature' in kwargs else None
        self.periods_in_use: bool = kwargs['periods_in_use'] if 'periods_in_use' in kwargs else None

def default_config():
    return FlashModelConfig(
        model_name = 'FlashCNN',
        input_station_count = 1,
        output_station_count = 1,
        input_variables_per_station = 6,
        output_variables_per_station = 6,
        lookback_days = 60,
        lookahead_days = 1,
        convolutional_kernel_size = 3,
        pooling_kernel_size = 2,
        convolutional_kernel_activation = 'ReLU',
        interlayer_channel_multiplier = 2,
        convolutional_layer_count = 2,
        dropout_rate = 0.2,
        input_channel_count = 1,
        fully_connected_activation = 'ReLU',
        use_average_pooling = False,
        fully_connected_layer_sizes = [512, 128],
        use_day_of_period_feature = True,
        periods_in_use = 1,
    )

class FlashModel(nn.Module):
    def __init__(self, config: FlashModelConfig):
        super(FlashModel, self).__init__()
        self.config = config
        self.model = None
        self._initialize()
    
    def _ensure_feasibility(self):
        h = self.config.lookback_days
        for _ in range(self.config.convolutional_layer_count):
            h += 1 - self.config.convolutional_kernel_size
            h //= self.config.pooling_kernel_size
        return h.is_integer()

    def _initialize(self):
        if not self._ensure_feasibility():
            raise ArithmeticError('Current model configuration results in non-integer parameters for model')

        layers = []
        c, cc, h, w = self.config.input_channel_count, 2*self.config.input_channel_count*self.config.interlayer_channel_multiplier, self.config.lookback_days, self.config.input_station_count*self.config.input_variables_per_station + (self.config.periods_in_use if self.config.use_day_of_period_feature else 0)

        layers.append(nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(1, w)))
        c, cc, h, w = cc, cc*self.config.interlayer_channel_multiplier, h, 1
        
        for _ in range(self.config.convolutional_layer_count):
            layers.append(nn.Conv2d(in_channels=c, out_channels=cc, kernel_size=(self.config.convolutional_kernel_size, 1)))
            c, cc, h, w = cc, cc*self.config.interlayer_channel_multiplier, h - self.config.convolutional_kernel_size + 1, 1
            if self.config.convolutional_kernel_activation == 'ReLU':
                layers.append(nn.ReLU())
            elif 'Leaky' in self.config.convolutional_kernel_activation:
                layers.append(nn.LeakyReLU(0.1))
            if self.config.use_average_pooling:
                layers.append(nn.AvgPool2d(kernel_size=(self.config.pooling_kernel_size, 1), stride=self.config.pooling_kernel_size))
            else:
                layers.append(nn.MaxPool2d(kernel_size=(self.config.pooling_kernel_size, 1), stride=self.config.pooling_kernel_size))

        layers.append(nn.Flatten())
        layers.append(nn.Dropout(p=self.config.dropout_rate))

        previous_layer_size = c * h
        for layer_size in self.config.fully_connected_layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_layer_size = layer_size
            if self.config.fully_connected_activation == 'ReLU':
                layers.append(nn.ReLU())
            elif 'Leaky' in self.config.fully_connected_activation:
                layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Linear(previous_layer_size, self.config.output_station_count*self.config.output_variables_per_station*self.config.lookahead_days))

        layers.append(nn.Unflatten(1, (self.config.lookahead_days, self.config.output_station_count*self.config.output_variables_per_station)))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)        
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to JSON config file for model')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to output file for model')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        loaded_config = json.load(file)

    config = FlashModelConfig(**loaded_config)
    model = FlashModel(config)

    with open(args.output_path, 'wb+') as file:
        pickle.dump(model, file)
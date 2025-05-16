import pandas as pd
from argparse import ArgumentParser
import json
import pickle
import torch

from work.normals import FlashNormals
from work.preprocessor import FlashPreprocessor


class FlashFeatureCreatorConfig:
    def __init__(self, **kwargs):
        self.include_lags_in_first_channel = kwargs['include_lags_in_first_channel'] if 'include_lags_in_first_channel' in kwargs else None
        self.included_lags = kwargs['included_lags'] if 'included_lags' in kwargs else None
        self.include_normals_in_first_channel = kwargs['include_normals_in_first_channel'] if 'include_normals_in_first_channel' in kwargs else None
        self.include_normals_as_second_channels = kwargs['include_normals_as_second_channels'] if 'include_normals_as_second_channels' in kwargs else None
        self.include_difference_channels = kwargs['include_difference_channels'] if 'include_difference_channels' in kwargs else None
        self.include_period_indicators_in_first_channel = kwargs['include_period_indicators_in_first_channel'] if 'include_period_indicators_in_first_channel' in kwargs else None
        self.include_sin_and_cos_channels = kwargs['include_sin_and_cos_channels'] if 'include_sin_and_cos_channels' in kwargs else None
        self.combine_sin_and_cos_channels = kwargs['combine_sin_and_cos_channels'] if 'combine_sin_and_cos_channels' in kwargs else None

class FlashFeatureCreator:
    def __init__(self, config: FlashFeatureCreatorConfig, preprocessor: FlashPreprocessor, normals: FlashNormals):
        self.config = config
        self.preprocessor = preprocessor
        self.normals = normals
    def create_from_date_range(self, start_date, end_date):
        # start with plain historical data
        historical = self.preprocessor[start_date:end_date]

        # create a tensor with size (1, H, W), where H is the number of days in the range and W is the number of station variables
        feature_tensor = torch.from_numpy(historical.values).float().unsqueeze(0)

        if self.config.include_normals_in_first_channel:
            for period_length in self.normals.period_lengths:
                n = self.normals.get_for_date_range('mean', period_length, start_date, end_date)
                normal_tensor = torch.from_numpy(n).float().unsqueeze(0)
                feature_tensor = torch.cat((feature_tensor, normal_tensor), 2)
        if self.config.include_lags_in_first_channel:
            for lag in self.config.included_lags:
                lagged_start = self.normals.get_for_date_range('mean', , start_date, end_date)



        # if self.config.include_normals_in_first_channel:
        #     for period_length in self.normals.period_lengths:
        #         range_normals = self.normals.get_for_date_range('mean', )

        if self.config.include_lags_in_first_channel:
            for lag in self.config.included_lags:
                lagged_historical =
import pandas as pd
import torch

class FeaturesCreationConfig:
    def __init__(self, **kwargs):
        self.stations = None
        self.station_variables = None
        self.variable_scales = None
        self.use_means_channel = None
        self.use_difference_channel = None
        self.use_squared_difference_channel = None
        self.time_variable_periods = None
        self.use_trig_time_variables = None
        self.use_time_transformed_features = None
        self.use_moving_average_features = None
        self.moving_averages_to_use = None
        self.use_lagged_features = None
        self.feature_lags_to_use = None

def create_normal_tensors(historical_data: pd.DataFrame):

def create_tensors(historical_data: pd.DataFrame):
    date_lookup = {}
    index = historical_data['DATE']
    for i in range(len(index)):
        date_lookup[index[i]] = i

    merged = pd.merge(historical_data, normal_data, on='DAY', how='left')
    mean_columns = [x for x in merged.columns if 'mean' in x]
    normals_for_historical = merged[['DATE', 'DAY'] + mean_columns]
    normals_for_historical.columns = historical_data.columns
    differences = historical_data.drop(['DATE', 'DAY'], axis=1).subtract(normals_for_historical.drop(['DATE', 'DAY'], axis=1))
    squared_differences = differences.pow(2)

    historical = historical_data.drop(['DATE', 'DAY'], axis=1)
    normal = normals_for_historical.drop(['DATE', 'DAY'], axis=1)

    h_tensors = torch.from_numpy(historical.values)
    n_tensors = torch.from_numpy(normal.values)
    d_tensors = torch.from_numpy(differences.values)
    s_tensors = torch.from_numpy(squared_differences.values)

    return torch.stack((h_tensors, n_tensors, d_tensors, s_tensors), dim=0), date_lookup
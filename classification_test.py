from model import FlashModel
from train import *
import pickle
import torch
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import os
from tqdm import tqdm
from forecast import *

# classifications
HTN = 'HTN' # higher than normal
NRM = 'NRM' # normal
LTN = 'LTN' # lower than normal

def get_period_classification(x, normal_x, margin):
    total = sum(x)
    total_normal = sum(normal_x)
    
    if total > total_normal * (1 + margin):
        return HTN
    if total < total_normal * (1 - margin):
        return LTN
    return NRM

def get_period_classifications(observed_variable, normal_variable, period_length, margin):
    num_periods = len(observed_variable) // period_length
    classifications = []
    for period in range(num_periods):
        period_begin = period * period_length
        period_end = (period + 1) * period_length
        classification = get_period_classification(observed_variable[period_begin:period_end], normal_variable[period_begin:period_end], margin)
        classifications.append(classification)
    return classifications
    
def get_forecasted_period_from_tensors(model, t, dl, forecast_start, forecast_length):
    forecast_start_idx = dl[forecast_start]
    forecast_end_idx = forecast_start_idx + forecast_length - 1

    actual_features = get_many_historical_features(forecast_start_idx, forecast_end_idx, t, dl, model.output_features, input=False)
    
    return actual_features

def classification_test(model_path, model_name, weeks, start_date, end_date, historical_data, normals_data, output_dir, margin):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    model.eval()
    
    historical = pd.read_csv(historical_data)
    normals = pd.read_csv(normals_data)
    normals.index = normals['DAY'].values

    t, dl = create_tensors(historical, normals)
    t = t.float()

    all_dates = pd.date_range(start=start_date, end=end_date)

    forecast_prcp_classifications = []
    forecast_tavg_classifications = []
    forecast_awnd_classifications = []

    observations_prcp_classifications = []
    observations_tavg_classifications = []
    observations_awnd_classifications = []
    
    for starting_date in tqdm(all_dates):
        days = 7 * weeks
        forecast, forecast_period_normals = create_forecast(model, historical, normals, str(starting_date.date()), days, False)
        actual_observations = get_forecasted_period_from_tensors(model, t, dl, str(starting_date.date()), days)

        station_idx = 0
        
        PRCP_forecast = forecast[:, tensor_idx('PRCP', station_idx)]
        TMAX_forecast = forecast[:, tensor_idx('TMAX', station_idx)]
        TMIN_forecast = forecast[:, tensor_idx('TMIN', station_idx)]
        TAVG_forecast = (TMAX_forecast + TMIN_forecast) / 2
        AWND_forecast = forecast[:, tensor_idx('AWND', station_idx)]

        PRCP_observations = actual_observations[:, tensor_idx('PRCP', station_idx)]
        TMAX_observations = actual_observations[:, tensor_idx('TMAX', station_idx)]
        TMIN_observations = actual_observations[:, tensor_idx('TMIN', station_idx)]
        TAVG_observations = (TMAX_observations + TMIN_observations) / 2
        AWND_observations = actual_observations[:, tensor_idx('AWND', station_idx)]

        PRCP_normal = forecast_period_normals[:, tensor_idx('PRCP', station_idx)]
        TMAX_normal = forecast_period_normals[:, tensor_idx('TMAX', station_idx)]
        TMIN_normal = forecast_period_normals[:, tensor_idx('TMIN', station_idx)]
        TAVG_normal = (TMAX_normal + TMIN_normal) / 2
        AWND_normal = forecast_period_normals[:, tensor_idx('AWND', station_idx)]

        forecast_PRCP_classifications = get_period_classifications(PRCP_forecast, PRCP_normal, 7, margin=margin)
        observations_PRCP_classifications = get_period_classifications(PRCP_observations, PRCP_normal, 7, margin=margin)
        forecast_prcp_classifications.append(forecast_PRCP_classifications)
        observations_prcp_classifications.append(observations_PRCP_classifications)
        
        forecast_TAVG_classifications = get_period_classifications(TAVG_forecast, TAVG_normal, 7, margin=margin)
        observations_TAVG_classifications = get_period_classifications(TAVG_observations, TAVG_normal, 7, margin=margin)
        forecast_tavg_classifications.append(forecast_TAVG_classifications)
        observations_tavg_classifications.append(observations_TAVG_classifications)

        forecast_AWND_classifications = get_period_classifications(AWND_forecast, AWND_normal, 7, margin=margin)
        observations_AWND_classifications = get_period_classifications(AWND_observations, AWND_normal, 7, margin=margin)
        forecast_awnd_classifications.append(forecast_AWND_classifications)
        observations_awnd_classifications.append(observations_AWND_classifications)

    forecast_prcp_classifications = pd.DataFrame(forecast_prcp_classifications)
    forecast_prcp_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_forecast_PRCP_classes.csv')
    forecast_prcp_classifications.to_csv(save_path, index=False)
    
    forecast_tavg_classifications = pd.DataFrame(forecast_tavg_classifications)
    forecast_tavg_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_forecast_TAVG_classes.csv')
    forecast_tavg_classifications.to_csv(save_path, index=False)
    
    forecast_awnd_classifications = pd.DataFrame(forecast_awnd_classifications)
    forecast_awnd_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_forecast_AWND_classes.csv')
    forecast_awnd_classifications.to_csv(save_path, index=False)

    observations_prcp_classifications = pd.DataFrame(observations_prcp_classifications)
    observations_prcp_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_observations_PRCP_classes.csv')
    observations_prcp_classifications.to_csv(save_path, index=False)
    
    observations_tavg_classifications = pd.DataFrame(observations_tavg_classifications)
    observations_tavg_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_observations_TAVG_classes.csv')
    observations_tavg_classifications.to_csv(save_path, index=False)
    
    observations_awnd_classifications = pd.DataFrame(observations_awnd_classifications)
    observations_awnd_classifications.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_observations_AWND_classes.csv')
    observations_awnd_classifications.to_csv(save_path, index=False)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help = "model checkpoint path")
    parser.add_argument("-n", "--model_name", type=str, required=True, help = "model name")
    parser.add_argument("-w", "--weeks", type=int, required=True, help = "weeks forward to forecast")
    parser.add_argument("--start_date", type=str, required=True, help = "testing start date")
    parser.add_argument("--end_date", type=str, required=True, help = "testing end date")
    parser.add_argument("--historical_data", type=str, required=True, help = "historical data filepath")
    parser.add_argument("--normals_data", type=str, required=True, help = "daily normals data filepath")
    parser.add_argument("--output_dir", type=str, required=True, help = "stress test results output directory")
    parser.add_argument("--margin", type=float, required=True, help = "margin for deviation from normal")
    args = parser.parse_args()

    classification_test(args.model_path,
                        args.model_name,
                        args.weeks,
                        args.start_date,
                        args.end_date,
                        args.historical_data,
                        args.normals_data,
                        args.output_dir,
                        args.margin)
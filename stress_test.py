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

def get_forecasted_period_from_tensors(model, t, dl, forecast_start, forecast_length):
    forecast_start_idx = dl[forecast_start]
    forecast_end_idx = forecast_start_idx + forecast_length - 1

    actual_features = get_many_historical_features(forecast_start_idx, forecast_end_idx, t, dl, model.output_features, input=False)
    
    return actual_features

def run_stress_test(model_path, model_name, weeks, start_date, end_date, historical_data, normals_data, output_dir):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    model.eval()
    
    historical = pd.read_csv(historical_data)
    normals = pd.read_csv(normals_data)
    normals.index = normals['DAY'].values

    t, dl = create_tensors(historical, normals)
    t = t.float()

    all_dates = pd.date_range(start=start_date, end=end_date)

    prcp_errors = []
    tavg_errors = []
    awnd_errors = []
    
    for starting_date in tqdm(all_dates):
        days = 7 * weeks
        forecast, _ = create_forecast(model, historical, normals, str(starting_date.date()), days, False)
        actual_observations = get_forecasted_period_from_tensors(model, t, dl, str(starting_date.date()), days)

        station_idx = 0
        
        PRCP = forecast[:, tensor_idx('PRCP', station_idx)]
        TMAX = forecast[:, tensor_idx('TMAX', station_idx)]
        TMIN = forecast[:, tensor_idx('TMIN', station_idx)]
        TAVG = (TMAX + TMIN) / 2
        AWND = forecast[:, tensor_idx('AWND', station_idx)]
        
        condensed_forecast = torch.stack((PRCP, TAVG, AWND), dim=1)

        PRCP = actual_observations[:, tensor_idx('PRCP', station_idx)]
        TMAX = actual_observations[:, tensor_idx('TMAX', station_idx)]
        TMIN = actual_observations[:, tensor_idx('TMIN', station_idx)]
        TAVG = (TMAX + TMIN) / 2
        AWND = actual_observations[:, tensor_idx('AWND', station_idx)]
        
        condensed_observations = torch.stack((PRCP, TAVG, AWND), dim=1)

        columns = [f'PRCP_{station_idx}', f'TAVG_{station_idx}', f'AWND_{station_idx}']

        forecast_df = pd.DataFrame(condensed_forecast.numpy())
        forecast_df.columns = columns
        forecast_df.insert(0, 'DAY', forecast_df.index)
    
        observations_df = pd.DataFrame(condensed_observations.numpy())
        observations_df.columns = columns
        observations_df.insert(0, 'DAY', observations_df.index)
        
        errors_df = observations_df.subtract(forecast_df).abs()
        
        prcp_errors.append(errors_df.loc[:, [f'PRCP_{station_idx}']].T)
        tavg_errors.append(errors_df.loc[:, [f'TAVG_{station_idx}']].T)
        awnd_errors.append(errors_df.loc[:, [f'AWND_{station_idx}']].T)

    all_prcp_errors_df = pd.concat(prcp_errors, ignore_index=True)
    all_prcp_errors_df.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_PRCP_error.csv')
    all_prcp_errors_df.to_csv(save_path, index=False)
    
    all_tavg_errors_df = pd.concat(tavg_errors, ignore_index=True)
    all_tavg_errors_df.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_TAVG_error.csv')
    all_tavg_errors_df.to_csv(save_path, index=False)
    
    all_awnd_errors_df = pd.concat(awnd_errors, ignore_index=True)
    all_awnd_errors_df.insert(0, 'FORECAST_START', all_dates)
    save_path = os.path.join(output_dir, f'{model_name}_AWND_error.csv')
    all_awnd_errors_df.to_csv(save_path, index=False)

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
    args = parser.parse_args()

    run_stress_test(args.model_path,
                    args.model_name,
                    args.weeks,
                    args.start_date,
                    args.end_date,
                    args.historical_data,
                    args.normals_data,
                    args.output_dir)
from model import FlashModel
from train import *
import pickle
import torch
import numpy as np
from argparse import ArgumentParser
import pandas as pd

def tensor_idx(variable_name, station_idx):
    match variable_name:
        case 'PRCP' | 'STORM':
            return 6*station_idx
        case 'TMAX':
            return 6*station_idx+1
        case 'TMIN':
            return 6*station_idx+2
        case 'AWND':
            return 6*station_idx+3
        case 'UWSF12':
            return 6*station_idx+4
        case 'VWSF12':
            return 6*station_idx+5

def derived_idx(variable_name, station_idx):
    match variable_name:
        case 'WSF12':
            return 7*station_idx
        case 'WDF12':
            return 7*station_idx+1
        case 'UWDF12':
            return 7*station_idx+2
        case 'VWDF12':
            return 7*station_idx+3
        case 'TAVG':
            return 7*station_idx+4
        case 'TSPREAD':
            return 7*station_idx+5
        case 'WSPREAD':
            return 7*station_idx+6

def display_forecast(forecast_tensor, forecast_date, station_idx, storm_mode):
    if storm_mode:
        STORM = float(forecast_tensor[tensor_idx('STORM', station_idx)])
    else:
        PRCP = float(forecast_tensor[tensor_idx('PRCP', station_idx)])
            
    TMAX = float(forecast_tensor[tensor_idx('TMAX', station_idx)])
    TMIN = float(forecast_tensor[tensor_idx('TMIN', station_idx)])
    AWND = float(forecast_tensor[tensor_idx('AWND', station_idx)])
    UWSF12 = float(forecast_tensor[tensor_idx('UWSF12', station_idx)])
    VWSF12 = float(forecast_tensor[tensor_idx('VWSF12', station_idx)])
    
    print('======================================')
    print(f'Forecast for {forecast_date} at station {station_idx}:')
    if storm_mode and STORM > 0:
        print(f'Expected to storm')
    elif storm_mode:
        print(f'Not expected to storm')
    else:
        print(f'Expected precipitation: {PRCP} hundredths in.')
    print(f'High temperature: {TMAX} deg F')
    print(f'Low temperature: {TMIN} deg F')
    print(f'Average wind speed: {AWND} mph')
    print(f'Eastward wind gust speed: {UWSF12} mph')
    print(f'Northward wind gust speed: {VWSF12} mph')
    print('======================================')

def normal_clamp_forecast(forecast_tensor, normals, num_stations, storm_mode):
    clamp = lambda x, min_val, max_val: min(max_val, max(x, min_val))
    clamped_forecast = torch.zeros_like(forecast_tensor)
    
    for station_idx in range(num_stations):
        if storm_mode:
            STORM = float(forecast_tensor[tensor_idx('STORM', station_idx)])
        else:
            PRCP = float(forecast_tensor[tensor_idx('PRCP', station_idx)])
    
        TMAX = float(forecast_tensor[tensor_idx('TMAX', station_idx)])
        TMIN = float(forecast_tensor[tensor_idx('TMIN', station_idx)])
        AWND = float(forecast_tensor[tensor_idx('AWND', station_idx)])
        UWSF12 = float(forecast_tensor[tensor_idx('UWSF12', station_idx)])
        VWSF12 = float(forecast_tensor[tensor_idx('VWSF12', station_idx)])

        if not storm_mode:
            PRCP_normal = float(normals[tensor_idx('PRCP', station_idx)])
        TMAX_normal = float(normals[tensor_idx('TMAX', station_idx)])
        TMIN_normal = float(normals[tensor_idx('TMIN', station_idx)])
        AWND_normal = float(normals[tensor_idx('AWND', station_idx)])
        UWSF12_normal = float(normals[tensor_idx('UWSF12', station_idx)])
        VWSF12_normal = float(normals[tensor_idx('VWSF12', station_idx)])

        if storm_mode:
            STORM = 100.0 if STORM > 50.0 else 0.0
        else:
            PRCP = clamp(PRCP, 0, 100 * PRCP_normal)
            
        TMAX = clamp(TMAX, TMIN + 1, 2 * abs(TMAX_normal))
        TMIN = clamp(TMIN, -2 * abs(TMIN_normal), TMAX - 1)
        AWND = clamp(AWND, 0, 4 * AWND_normal)
        UWSF12 = clamp(UWSF12, -4 * abs(UWSF12), 4 * abs(UWSF12))
        VWSF12 = clamp(VWSF12, -4 * abs(VWSF12), 4 * abs(VWSF12))

        if storm_mode:
            clamped_forecast[tensor_idx('STORM', station_idx)] = STORM
        else:
            clamped_forecast[tensor_idx('PRCP', station_idx)] = PRCP
        
        clamped_forecast[tensor_idx('TMAX', station_idx)] = TMAX
        clamped_forecast[tensor_idx('TMIN', station_idx)] = TMIN
        clamped_forecast[tensor_idx('AWND', station_idx)] = AWND
        clamped_forecast[tensor_idx('UWSF12', station_idx)] = UWSF12
        clamped_forecast[tensor_idx('VWSF12', station_idx)] = VWSF12

    return clamped_forecast

def create_derived_values_from_forecast(forecast_tensor, forecast_date, num_stations, out_features):
    derived_tensor = torch.zeros(out_features - forecast_tensor.size()[0] - 3)

    for station_idx in range(num_stations):
        #PRCP = float(forecast_tensor[tensor_idx('PRCP', station_idx)])
        TMAX = float(forecast_tensor[tensor_idx('TMAX', station_idx)])
        TMIN = float(forecast_tensor[tensor_idx('TMIN', station_idx)])
        AWND = float(forecast_tensor[tensor_idx('AWND', station_idx)])
        UWSF12 = float(forecast_tensor[tensor_idx('UWSF12', station_idx)])
        VWSF12 = float(forecast_tensor[tensor_idx('VWSF12', station_idx)])

        WDF12 = np.arctan(VWSF12 / UWSF12)
        WSF12 = UWSF12 / np.cos(WDF12)
        UWDF12 = np.cos(WDF12) * np.cos(WDF12)
        VWDF12 = np.sin(WDF12) * np.sin(WDF12)
        TAVG = (TMAX + TMIN) / 2
        TSPREAD = TMAX - TMIN
        WSPREAD = WSF12 - AWND

        derived_tensor[derived_idx('WDF12', station_idx)] = WDF12
        derived_tensor[derived_idx('WSF12', station_idx)] = WSF12
        derived_tensor[derived_idx('UWDF12', station_idx)] = UWDF12
        derived_tensor[derived_idx('VWDF12', station_idx)] = VWDF12
        derived_tensor[derived_idx('TAVG', station_idx)] = TAVG
        derived_tensor[derived_idx('TSPREAD', station_idx)] = TSPREAD
        derived_tensor[derived_idx('WSPREAD', station_idx)] = WSPREAD
    
    DAY365 = pd.to_datetime(forecast_date).day_of_year / 365.25
    SIN = np.sin(2 * np.pi * DAY365)
    COS = np.cos(2 * np.pi * DAY365)
    
    time_tensor = torch.tensor((DAY365, SIN, COS))
    return torch.concat((forecast_tensor, derived_tensor, time_tensor), dim=0)

def fetch_normals_for_date(normals, forecast_date):
    mean_columns = [x for x in normals.columns if 'mean' in x]
    day = forecast_date.day_of_year
    day = day if day < 366 else 1
    means = normals[mean_columns].loc[day]
    return torch.from_numpy(means.values)

def create_forecast(model, historical, normals, forecast_start, forecast_length, storm_mode):
    t, dl = create_tensors(historical, normals)
    
    forecast_start_idx = dl[forecast_start]
    historical_data_start = forecast_start_idx - model.sequence_length
    historical_data_end = forecast_start_idx - 1
    
    current_lookback = get_many_historical_features(historical_data_start, historical_data_end, t.float(), dl, model.output_features, input=True)
    #print(current_lookback.size())
    
    forecast = []

    start_date = pd.to_datetime(forecast_start)
    for i in range(forecast_length):
        current_date = start_date + pd.Timedelta(days=i)
        prediction = model(current_lookback.unsqueeze(0))[0]
        
        daily_means = fetch_normals_for_date(normals, current_date)
        clamped_prediction = normal_clamp_forecast(prediction, daily_means, model.stations, storm_mode)
        forecast.append(clamped_prediction)
        
        extended_prediction = create_derived_values_from_forecast(clamped_prediction, current_date, model.stations, current_lookback.size()[2])
        difference = extended_prediction - daily_means
        squared = difference * difference

        features = torch.stack((extended_prediction.unsqueeze(0), 
                                daily_means.unsqueeze(0), 
                                difference.unsqueeze(0), 
                                squared.unsqueeze(0)), dim=0)

        current_lookback = torch.concat((current_lookback[:, 1:], features.float()), dim=1)

    return torch.stack(forecast, dim=0)

def get_forecasted_period_from_historical(model, historical, normals, forecast_start, forecast_length):
    t, dl = create_tensors(historical, normals)
    
    forecast_start_idx = dl[forecast_start]
    forecast_end_idx = forecast_start_idx + forecast_length - 1

    actual_features = get_many_historical_features(forecast_start_idx, forecast_end_idx, t.float(), dl, model.output_features, input=False)
    
    return actual_features

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help = "model checkpoint path")
    parser.add_argument("-w", "--weeks", type=int, required=True, help = "weeks forward to forecast")
    parser.add_argument("--start_date", type=str, required=True, help = "forecast start date")
    parser.add_argument("--historical_data", type=str, required=True, help = "historical data filepath")
    parser.add_argument("--normals_data", type=str, required=True, help = "daily normals data filepath")
    parser.add_argument("--forecast_path", type=str, required=True, help = "forecast output filepath")
    parser.add_argument("--observations_path", type=str, required=True, help = "actual observations filepath")
    parser.add_argument("--storm_mode", action='store_true', help = "storm mode")
    parser.add_argument("--condense", action='store_true', help = "condense to only output PRCP/STORM, TAVG, and AWND")
    args = parser.parse_args()
    
    with open(args.model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    model.eval()
    
    historical = pd.read_csv(args.historical_data)
    normals = pd.read_csv(args.normals_data)
    normals.index = normals['DAY'].values
    
    days = 7 * args.weeks
    forecast = create_forecast(model, historical, normals, args.start_date, days, args.storm_mode)
    actual_observations = get_forecasted_period_from_historical(model, historical, normals, args.start_date, days)

    if args.condense:
        condensed_station_forecasts = []
        condensed_station_observations = []
        
        for station_idx in range(model.stations):
            
            STORM = forecast[:, tensor_idx('STORM', station_idx)]
            PRCP = forecast[:, tensor_idx('PRCP', station_idx)]
            TMAX = forecast[:, tensor_idx('TMAX', station_idx)]
            TMIN = forecast[:, tensor_idx('TMIN', station_idx)]
            TAVG = (TMAX + TMIN) / 2
            AWND = forecast[:, tensor_idx('AWND', station_idx)]
            
            condensed_station_forecast = torch.stack((STORM if args.storm_mode else PRCP, TAVG, AWND), dim=1)
            condensed_station_forecasts.append(condensed_station_forecast)

            STORM = actual_observations[:, tensor_idx('STORM', station_idx)]
            PRCP = actual_observations[:, tensor_idx('PRCP', station_idx)]
            TMAX = actual_observations[:, tensor_idx('TMAX', station_idx)]
            TMIN = actual_observations[:, tensor_idx('TMIN', station_idx)]
            TAVG = (TMAX + TMIN) / 2
            AWND = actual_observations[:, tensor_idx('AWND', station_idx)]
            
            condensed_station_observation = torch.stack((STORM if args.storm_mode else PRCP, TAVG, AWND), dim=1)
            condensed_station_observations.append(condensed_station_observation)

        forecast = torch.cat(condensed_station_forecasts, dim=1)
        actual_observations = torch.cat(condensed_station_observations, dim=1)
        
        columns = []
        for station_idx in range(model.stations):
            columns.extend([f'STORM_{station_idx}' if args.storm_mode else f'PRCP_{station_idx}', 
                            f'TAVG_{station_idx}', 
                            f'AWND_{station_idx}'])
    else:
        columns = []
        for station_idx in range(model.stations):
            columns.extend([f'STORM_{station_idx}' if args.storm_mode else f'PRCP_{station_idx}', 
                            f'TMAX_{station_idx}',
                            f'TMIN_{station_idx}',
                            f'AWND_{station_idx}',
                            f'UWSF12_{station_idx}',
                            f'VWSF12_{station_idx}'])

    forecast_df = pd.DataFrame(forecast.numpy())
    forecast_df.columns = columns
    forecast_df.insert(0, 'DAY', forecast_df.index)
    forecast_df.insert(0, 'DATE', pd.date_range(start=args.start_date, periods=len(forecast_df), freq='D'))

    observations_df = pd.DataFrame(actual_observations.numpy())
    observations_df.columns = columns
    observations_df.insert(0, 'DAY', observations_df.index)
    observations_df.insert(0, 'DATE', pd.date_range(start=args.start_date, periods=len(observations_df), freq='D'))

    forecast_df.to_csv(args.forecast_path, index=False)
    observations_df.to_csv(args.observations_path, index=False)      
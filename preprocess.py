import pandas as pd
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True, help = "raw data filepath")
    parser.add_argument("--out_path", type=str, required=True, help = "output features filepath")
    parser.add_argument("--normal_path", type=str, required=True, help = "normals filepath")
    parser.add_argument("--storm_mode", type=int, required=False, default=0, help = "normals filepath")
    parser.add_argument("--stats_path", type=str, required=False, default='storm_statistics.csv', help = "storm statistics filepath")
    args = parser.parse_args()
    
    df = pd.read_csv(args.raw_path)
    df.set_index('DATE', inplace=True)
    
    stations = df['STATION'].unique()
    for station in stations:
        if df[df['STATION'] == station]['PRCP'].sum() == 0:
            df = df[df['STATION'] != station]
    stations = df['STATION'].unique()
    
    print(stations)
    
    df = df[['STATION', 'PRCP', 'TMAX', 'TMIN', 'AWND', 'WDF1', 'WDF2', 'WSF1', 'WSF2']]
    df = df.pivot(columns='STATION', values=['PRCP', 'TMAX', 'TMIN', 'AWND', 'WDF1', 'WDF2', 'WSF1', 'WSF2'])
    df = df.loc['1984-01-01':'2024-12-31']
    df.columns = ['_'.join(col) for col in df.columns]

    if bool(args.storm_mode):
        all_storms = []
        for station in stations:
            prcp_col = f'PRCP_{station}'
            storm_col = f'STORM_{station}'
            
            df[storm_col] = df[prcp_col].apply(lambda x: 1 if x > 0 else 0)
            
            PRCPs = df[prcp_col].values
            storms = []
        
            current_storm_length = 0
            current_storm_prcp = 0
            for amount in PRCPs:
                if amount > 0:
                    current_storm_length += 1
                    current_storm_prcp += float(amount)
                elif current_storm_length > 0:
                    storms.append((current_storm_length, current_storm_prcp))
                    current_storm_length = 0
                    current_storm_prcp = 0
            all_storms.extend(storms)

            df.drop(prcp_col, axis=1)
        
        storm_stats = {}
        for storm_length in range(100):
            day_storms = [amount for length, amount in all_storms if length == storm_length]
            if len(day_storms) < 100:
                continue
            mean = np.mean(day_storms)
            std = np.std(day_storms)
            storm_stats[storm_length] = {
                'Count': len(day_storms),
                'Mean': float(mean),
                'Std': float(std),
            }
        
        storm_df = pd.DataFrame(storm_stats).T
        storm_df.to_csv(args.stats_path)
    
    df = df.ffill().fillna(0)
    
    for station in stations:
        df[f'WSF12_{station}'] = df[f'WSF1_{station}'] + df[f'WSF2_{station}']
        df[f'WDF12_{station}'] = np.pi * (df[f'WDF1_{station}'] + df[f'WDF2_{station}']) / 2
        df[f'UWSF12_{station}'] = df[f'WSF12_{station}'] * np.cos(df[f'WDF12_{station}'])
        df[f'VWSF12_{station}'] = df[f'WSF12_{station}'] * np.sin(df[f'WDF12_{station}'])
        df[f'UWDF12_{station}'] = np.cos(df[f'WDF12_{station}']).pow(2)
        df[f'VWDF12_{station}'] = np.sin(df[f'WDF12_{station}']).pow(2)
        df = df.drop([f'WDF1_{station}', f'WDF2_{station}', f'WSF1_{station}', f'WSF2_{station}'], axis=1)
    
        df[f'TAVG_{station}'] = (df[f'TMAX_{station}'] + df[f'TMIN_{station}'])/2
        df[f'TSPREAD_{station}'] = (df[f'TMAX_{station}'] - df[f'TMIN_{station}'])
        df[f'WSPREAD_{station}'] = df[f'WSF12_{station}'] - df[f'AWND_{station}']
    
    new_column_order = []
    for station in stations:
        if bool(args.storm_mode):
            new_column_order += [f'PRCP_{station}', f'TMAX_{station}', f'TMIN_{station}', f'AWND_{station}', f'UWSF12_{station}', f'VWSF12_{station}']
        else:
            new_column_order += [f'STORM_{station}', f'TMAX_{station}', f'TMIN_{station}', f'AWND_{station}', f'UWSF12_{station}', f'VWSF12_{station}']
    for station in stations:
        new_column_order += [f'WSF12_{station}', f'WDF12_{station}', f'UWDF12_{station}', f'VWDF12_{station}', f'TAVG_{station}', f'TSPREAD_{station}', f'WSPREAD_{station}']
    df = df[new_column_order]
    
    df.insert(0, 'DAY', pd.to_datetime(df.index).day_of_year)
    df['DAY'] = df['DAY'].apply(lambda x: x if x < 366 else 1)
    df['DAY365'] = df['DAY'] / 365.25
    df['SIN'] = np.sin(2 * np.pi * df['DAY365'])
    df['COS'] = np.cos(2 * np.pi * df['DAY365'])
    
    df.to_csv(args.out_path)
    
    limited = df.loc[:'2000-01-01']
    
    normals = limited.groupby('DAY').agg(['mean', 'std'])
    normals.columns = ['_'.join(col) for col in normals.columns]
    normals.to_csv(args.normal_path)
import pandas as pd
from argparse import ArgumentParser
import json
import pickle

class FlashPreprocessorConfig:
    def __init__(self, **kwargs):
        self.stations = kwargs['stations'] if 'stations' in kwargs else None
        self.station_variables = kwargs['station_variables'] if 'station_variables' in kwargs else None
        self.variable_scales = kwargs['variable_scales'] if 'variable_scales' in kwargs else None

class FlashPreprocessor:
    def __init__(self, config: FlashPreprocessorConfig, historical_filepath: str):
        self.config = config
        self.preprocessed = None
        self._initialize(historical_filepath)
    def _initialize(self, historical_filepath: str):
        historical_df = pd.read_csv(historical_filepath)

        historical_df['DATE'] = pd.to_datetime(historical_df['DATE'])
        historical_df.set_index('DATE', inplace=True)

        # trim the dataframe to only have the specified stations and variables
        historical_df = historical_df[historical_df['STATION'].isin(self.config.stations)]
        historical_df = historical_df[['STATION' + self.config.station_variables]]

        # scale each variable appropriately, as requested
        for variable in self.config.station_variables:
            if variable in self.config.variable_scales:
                scale = self.config.variable_scales[variable]
                historical_df[variable] = historical_df[variable] * scale

        # pivot so that each station-variable combo has its own column, named like {variable}_{station}
        historical_df = historical_df.pivot(columns='STATION', values=self.config.station_variables)
        historical_df.columns = ['_'.join(col) for col in historical_df.columns]

        # now fill in any missing data that can be filled in, and all the rest set to 0.0
        historical_df = historical_df.ffill().fillna(0.0)

        # add year and day of the year column
        #historical_df.insert(0, 'YEAR', pd.to_datetime(historical_df.index).year)
        #historical_df.insert(1, 'DAY_OF_YEAR', pd.to_datetime(historical_df.index).day_of_year)

        self.preprocessed = historical_df

    def __getitem__(self, idx):
        if isinstance(idx, int) or (isinstance(idx, slice) and isinstance(idx.start, int)):
            return self.preprocessed.iloc[idx]
        return self.preprocessed.loc[idx]
    def get_for_date_range(self, start_date, end_date, lag=0):
        start_datetime = pd.to_datetime(start_date) - pd.Timedelta(days=lag)
        end_datetime = pd.to_datetime(end_date) - pd.Timedelta(days=lag)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to JSON config file for model')
    parser.add_argument('-r', '--raw_path', type=str, required=True, help='Path to raw data file')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output path for initialized preprocessor')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        loaded_config = json.load(file)

    config = FlashPreprocessorConfig(**loaded_config)
    p = FlashPreprocessor(config, args.raw_path)

    with open(args.output_path, 'wb+') as file:
        pickle.dump(p, file)
import pandas as pd
from argparse import ArgumentParser
import json
import pickle
from work.preprocessor import FlashPreprocessor

def convert_to_day_of_period(date, reference_date, period_days):
    delta = date - reference_date
    return 1 + delta.days % period_days

class FlashNormalsConfig:
    def __init__(self, **kwargs):
        self.start_date = kwargs['start_date'] if 'start_date' in kwargs else None
        self.end_date = kwargs['end_date'] if 'end_date' in kwargs else None
        self.period_lengths = kwargs['variable_scales'] if 'variable_scales' in kwargs else None

class FlashNormals:
    def __init__(self, config: FlashNormalsConfig, preprocessor: FlashPreprocessor):
        self.config = config
        self.reference = pd.to_datetime(self.config.start_date)
        self.normals = None
        self._initialize(preprocessor)
    def _initialize(self, preprocessor: FlashPreprocessor):
        sample = preprocessor[self.config.start_date:self.config.end_date]

        self.normals = {}
        for period_length in self.config.period_lengths:
            sample.insert(0, 'DAY', sample.index.apply(lambda x: convert_to_day_of_period(x, self.reference, period_length)))
            grouped = sample.groupby('DAY')
            means = grouped.mean()
            maxes = grouped.max()
            mins = grouped.min()
            self.normals[period_length] = {
                'mean': means.mean(),
                'max': maxes.max(),
                'min': mins.min()
            }
            sample.drop('DAY', axis=1, inplace=True)
    def get(self, normal_type, period_length, day_of_period):
        return self.normals[period_length][normal_type].loc[day_of_period]
    def get_for_date(self, normal_type, period_length, date):
        day_of_period = convert_to_day_of_period(pd.to_datetime(date), self.reference, period_length)
        return self.get(normal_type, period_length, day_of_period)
    def get_for_date_range(self, normal_type, period_length, start_date, end_date):
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

        all_normals = []
        days_between = (end_datetime - start_datetime).days

        for i in range(days_between+1):
            current_datetime = start_datetime + pd.Timedelta(days=i)
            current_day = convert_to_day_of_period(current_datetime, self.reference, period_length)
            current_day_normals = self.get(normal_type, period_length, current_day)
            all_normals.append(current_day_normals)

        return pd.concat(all_normals)
    def get_day_of_period(self, period_length, date):
        return convert_to_day_of_period(pd.to_datetime(date), self.reference, period_length)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to JSON config file for model')
    parser.add_argument('-p', '--preprocessor_path', type=str, required=True, help='Path to initialized preprocessor')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output path for initialized normals')

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        loaded_config = json.load(file)

    with open(args.preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)

    config = FlashNormalsConfig(**loaded_config)
    normals = FlashNormals(config, preprocessor)

    with open(args.preprocessor_path, 'wb+') as file:
        pickle.dump(normals, file)
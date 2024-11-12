# Holds the config for all code
import json
import os
from datetime import datetime
import pandas as pd

_base_path = "/Users/stephanie/src/N-LASR-model"
_data_dir = os.path.join(_base_path, 'data')
raw_price_file_post_fix = '_1hr_historical_data_final'

base_config = {
    "name": "base",

    # tickers to use
    "tickers": [
        "AAPL", "ADBE", "AIG", "ALL", "AMD", "AMAT", "AMGN", "AMZN", "AON",
        "APD", "AVGO", "AXP", "BAC", "BABA", "BAX", "BBY", "BDX", "BIIB", "BLK",
        "BMY", "BK", "CB", "C", "CHD", "CI", "CINF", "CL", "CLX", "CMG",
        "CRM", "COST", "CRWD", "CSCO", "CVS", "DDOG", "DG", "DHR", "DOCU", "DPZ",
        "DRI", "EW", "ECL", "F", "FDX", "FMC", "GM", "GILD", "GIS", "GOOG",
        "GS", "HD", "HOG", "HUBS", "IBM", "IFF", "ILMN", "INTC", "INTU", "ISRG",
        "JNJ", "JPM", "K", "KHC", "KR", "LNC", "LOW", "LULU", "MA", "MDB",
        "MCD", "MDT", "META", "MKC", "MMM", "MSFT", "MU", "NET", "NFLX", "NOW",
        "NVDA", "OKTA", "ORCL", "PEP", "PG", "PGR", "PFE", "PLTR", "PNC", "PPG",
        "PRU", "PYPL", "REGN", "ROKU", "RNG", "RL", "SBUX", "SCHW", "SHOP", "SKX",
        "SNOW", "SPGI", "SPY", "SQ", "SYK", "T", "TEAM", "TMO", "TRV", "TSCO",
        "TSLA", "TXN", "TROW", "TWLO", "UA", "UAA", "UNH", "USB", "V", "VRTX",
        "WBA", "WDAY", "WFC", "WMT", "YUM", "ZBH", "ZM"
    ],

    # data resampling and filtering
    "resample_interval": '30min',
    "exchange_start_time": '14:30:00',
    "exchange_end_time": '20:30:00',

    # targets
    "opening_trade_day_of_week": 0,  # Monday = 0
    "closing_trade_num_days_later": 4,
    "target_num_bins": 5,  # number of bins for target labels

    # train / test split
    "test_size": 0.7,
    "random_state": 42,
    "shuffle": False
}

def get_raw_prices_directory():
    raw_prices_directory = os.path.join(_data_dir, 'raw_prices')
    os.makedirs(raw_prices_directory, exist_ok=True)
    return raw_prices_directory

def _get_config_file_path(config_name):
    config_directory = os.path.join(_data_dir, 'config')
    os.makedirs(config_directory, exist_ok=True)
    return os.path.join(config_directory, config_name + '.json')


def _write_config(config):
    with open(_get_config_file_path(config['name']), 'w') as json_file:
        # noinspection PyTypeChecker
        json.dump(config, json_file, indent=4)

def read_config(config_name):
    with open(_get_config_file_path(config_name), 'r') as json_file:
        config = json.load(json_file)
    config['exchange_start_time'] = pd.to_datetime(config['exchange_start_time']).time(),
    config['exchange_end_time'] = pd.to_datetime(config['exchange_end_time']).time(),

    return config

def _get_model_dir_path(config_name):
    model_directory = os.path.join(_data_dir, 'model')
    os.makedirs(model_directory, exist_ok=True)
    return model_directory

def _write_model_file(config_name, file_name, targets):
    targets_file_path = (
        _get_model_dir_path(config_name) +
        config_name + '_' +
        datetime.now().replace(microsecond=0).isoformat() + '_' + file_name + '.pkl'
    )
    print('writing file to:', targets_file_path)
    targets.to_pickle(targets_file_path)

def write_targets_per_stock_by_date(config_name, targets):
    _write_model_file(config_name, 'targets_per_stock_by_date', targets)

def main():
    _write_config(base_config)

if __name__ == "__main__":
    main()
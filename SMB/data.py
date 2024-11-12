import json
import datetime
import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)


#
# Static Data
#
base_path = "/Users/stephanie/src/N-LASR-model"
data_dir = os.path.join(base_path, 'data')
raw_price_file_post_fix = '_1hr_historical_data_final'

#
# Config
#
def _get_config_file_path(config_name):
    config_directory = os.path.join(data_dir, 'config')
    os.makedirs(config_directory, exist_ok=True)
    config_file = os.path.join(config_directory, config_name + '.json')
    if not os.path.exists(config_file):
        msg = "Path does not exist: %s", config_file
        logger.error(msg)
        raise ValueError(msg)
    return config_file


def _write_config(config):
    with open(_get_config_file_path(config['name']), 'w') as json_file:
        logger.info("Writing config: %s", config['name'])
        # noinspection PyTypeChecker
        json.dump(config, json_file, indent=4)


def read_config(config_name):
    with open(_get_config_file_path(config_name), 'r') as json_file:
        logger.info("Reading config: %s", config_name)
        config = json.load(json_file)
    config['exchange_start_time'] = pd.to_datetime(config['exchange_start_time']).time()
    config['exchange_end_time'] = pd.to_datetime(config['exchange_end_time']).time()

    return config

#
# Prices
#
def get_raw_prices_directory():
    raw_prices_directory = os.path.join(data_dir, 'raw_prices')
    if not os.path.isdir(raw_prices_directory):
        msg = "Path does not exist: %s", raw_prices_directory
        logger.error(msg)
        raise ValueError(msg)
    return raw_prices_directory

# Reads a single data file, sets columns and index
def read_prices_file(ticker):
    file_path = os.path.join(get_raw_prices_directory(), f'{ticker}{raw_price_file_post_fix}.csv')    # Read the CSV file into a DataFrame
    logger.info("Reading prices file: %s", file_path)
    df = pd.read_csv(file_path)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
    # explicitly set the date format to avoid confusion with days before months
    df.set_index(pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S'), inplace=True)
    df.sort_index(inplace=True) # sort as file data may be out of order

    return df


# Modifies input data.
# Resamples to be at 'resample_interval', removes weekends and times outside exchange hours
def resample_filter_data(data, resample_interval, exchange_start_time, exchange_end_time):
    try:
        resampled_data = data.resample(resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'barCount': 'sum'
        })
    except KeyError as e:
        logger.error("Missing column in data: %s", e)
        return None
    logger.debug(resampled_data.iloc[:10, :])
    logger.debug(resampled_data.isna().sum())
    logger.debug(resampled_data.shape)

    # remove weekends
    hourly_data = resampled_data[resampled_data.index.dayofweek < 5]
    logger.debug(hourly_data.shape)

    # remove exchange non-trading hours
    # noinspection PyTypeChecker
    if not isinstance(exchange_start_time, datetime.time) or not isinstance(exchange_end_time, datetime.time):
        msg = "exchange_start_time and exchange_end_time must be datetime.time instances."
        logger.error(msg)
        raise ValueError(msg)
    if exchange_start_time is None:
        msg = "exchange_start_time must be specified."
        logger.error(msg)
        raise ValueError(msg)
    if exchange_end_time is None:
        msg = "exchange_end_time must be specified."
        logger.error(msg)
        raise ValueError(msg)
    filtered_data = hourly_data[
        (hourly_data.index.time >= exchange_start_time) &
        (hourly_data.index.time <= exchange_end_time)
        ]
    # time of day is removed by pandas as three is only one entry per day
    logger.debug(filtered_data.shape)
    logger.debug(filtered_data)

    return filtered_data

#
# Model Data
#
def _get_model_dir_path(config_name):
    model_directory = os.path.join(data_dir, 'model', config_name)
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


def _write_model_file(config_name, file_name, targets):
    file_path = os.path.join(_get_model_dir_path(config_name), file_name)
    logger.info('Writing model file to: %s', file_path)
    targets.to_pickle(file_path)


def _read_model_file(config_name, file_name):
    file_path = os.path.join(_get_model_dir_path(config_name), file_name)
    if not os.path.exists(file_path):
        msg = "Path does not exist:", file_path
        logger.error(msg)
        raise ValueError(msg)
    logger.info('Reading model file: %s', file_path)
    return pd.read_pickle(file_path)


def get_targets_per_stock_by_date_filename():
    return 'targets_per_stock_by_date' + '_' + datetime.datetime.now().replace(microsecond=0).isoformat() + '.pkl'


def write_targets_per_stock_by_date(config_name, targets):
    file_name = get_targets_per_stock_by_date_filename()
    _write_model_file(config_name, file_name, targets)
    return file_name


def read_targets_per_stock_by_date(config_name, file_name):
    return _read_model_file(config_name, file_name)

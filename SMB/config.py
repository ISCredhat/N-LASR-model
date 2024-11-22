# Holds the config for all code
import json
import pandas as pd
from SMB.data import _get_config_file_path
from my_logger import get_logger

logger = get_logger()


NAME = 'name'
TICKERS = 'tickers'
DATA_RESAMPLE_INTERVAL = 'data_resample_interval'
DATA_EXCHANGE_START_TIME = 'data_exchange_start_time'
DATA_EXCHANGE_END_TIME = 'data_exchange_end_time'
LIMIT_DROP_NA_ROWS = 'limit_drop_na_rows'
FEATURES_INDICATOR_ROLLING_WINDOW = 'features_indicator_rolling_window'
FEATURES_NUM_BINS = 'feature_num_bins'
FEATURES_TARGETS_INDEX_DIFFERENCE_LIMIT = 'features_targets_index_difference_limit'
TARGETS_OPEN_TRADE_DAY_OF_WEEK = 'targets_opening_trade_day_of_week'
TARGETS_CLOSING_TRADE_NUM_DAYS_LATER = 'targets_closing_trade_num_days_later'
TARGETS_NUM_BINS = 'targets_num_bins'
TRAIN_TEST_SPLIT = 'train_test_split'
TRAIN_TEST_RANDOM_STATE = 'train_test_random_state'
TRAIN_TEST_SHUFFLE = 'train_test_shuffle'

base_config = {
    "name": "base",

    # tickers to use
    TICKERS: [
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
    DATA_RESAMPLE_INTERVAL: '30min',
    DATA_EXCHANGE_START_TIME: '14:30:00',
    DATA_EXCHANGE_END_TIME: '20:30:00',

    # features and targets
    LIMIT_DROP_NA_ROWS: 8,

    # features
    FEATURES_INDICATOR_ROLLING_WINDOW: 200, # indicators
    FEATURES_NUM_BINS: 8,  # number of bins for target labels
    FEATURES_TARGETS_INDEX_DIFFERENCE_LIMIT: 2,

    # targets
    TARGETS_OPEN_TRADE_DAY_OF_WEEK: 0,  # Monday = 0
    TARGETS_CLOSING_TRADE_NUM_DAYS_LATER: 4,
    TARGETS_NUM_BINS: 5,  # number of bins for target labels

    # train / test split
    TRAIN_TEST_SPLIT: 0.7,
    TRAIN_TEST_RANDOM_STATE: 42,
    TRAIN_TEST_SHUFFLE: False
}


def _write_config(config):
    with open(_get_config_file_path(config[NAME]), 'w') as json_file:
        logger.info("Writing config: %s", config[NAME])
        # noinspection PyTypeChecker
        json.dump(config, json_file, indent=4)


def read_config(config_name):
    with open(_get_config_file_path(config_name), 'r') as json_file:
        logger.info("Reading config: %s", config_name)
        config = json.load(json_file)
    config[DATA_EXCHANGE_START_TIME] = pd.to_datetime(config[DATA_EXCHANGE_START_TIME]).time()
    config[DATA_EXCHANGE_END_TIME] = pd.to_datetime(config[DATA_EXCHANGE_END_TIME]).time()

    return config

#
# run this to generate configs and save to file
#
def main():
    _write_config(base_config)


if __name__ == "__main__":
    main()




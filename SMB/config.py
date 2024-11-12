# Holds the config for all code

from SMB.data import _write_config

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


def main():
    _write_config(base_config)

if __name__ == "__main__":
    main()
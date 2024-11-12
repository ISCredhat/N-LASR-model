import pandas as pd
import os
from SMB.config import get_raw_prices_directory, raw_price_file_post_fix


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
        print(f"Missing column in data: {e}")
        return None
    print(resampled_data.iloc[:10,:])
    print(resampled_data.isna().sum())
    print(resampled_data.shape)

    # remove weekends
    hourly_data = resampled_data[resampled_data.index.dayofweek < 5]
    print(hourly_data.shape)

    # remove exchange non-trading hours
    if exchange_start_time is None:
        raise ValueError("exchange_start_time ust be specified.")
    if exchange_end_time is None:
        raise ValueError("exchange_end_time ust be specified.")
    filtered_data = hourly_data[
        (hourly_data.index.time >= exchange_start_time) &
        (hourly_data.index.time <= exchange_end_time)
        ]
    # time of day is removed by pandas as three is only one entry per day
    print(filtered_data.shape)
    print(filtered_data)

    return filtered_data

# Reads a single data file, sets columns and index
def read_data(ticker):
    file_path = os.path.join(get_raw_prices_directory(), f'{ticker}{raw_price_file_post_fix}.csv')    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
    # explicitly set the date format to avoid confusion with days before months
    df.set_index(pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S'), inplace=True)
    df.sort_index(inplace=True) # sort as file data may be out of order

    return df
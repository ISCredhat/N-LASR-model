import pandas as pd
from SMB.data import read_prices_file, resample_filter_data


def calc_returns(ticker, config):
    try:
        # Ensure config has all required fields
        required_keys = ['resample_interval', 'exchange_start_time', 'exchange_end_time', 'opening_trade_day_of_week',
                         'closing_trade_num_days_later']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing '{key}' in configuration.")

        #
        # get raw data from file
        #
        data = read_prices_file(ticker)
        # print(data.columns)
        # print(data.dtypes)
        print(data.shape)

        #
        # resample to 'resample_interval', remove weekends and times outside exchange hours
        #
        resampled_filtered_data = resample_filter_data(data, config['resample_interval'], config['exchange_start_time'], config['exchange_end_time'])
        # print(data.columns)
        # print(data.dtypes)
        print(data.shape)

        # it may be possible to do a forward and backwards fill here
        # data_fwd_fill = resampled_filtered_data.ffill()
        # print(data_fwd_fill.iloc[:100, :])
        # print(data_fwd_fill.isna().sum())
        # print(data_fwd_fill.shape)

        #
        # For the OPENING & CLOSING trades we use back-filled prices so we can't trade in the past
        # and will only trade at a price available at the reference time or later
        #
        data_back_fill = resampled_filtered_data.bfill()
        print(data_back_fill.iloc[:100, :])
        print(data_back_fill.isna().sum())
        print(data_back_fill.shape)

        # Condition to raise informative error
        if data_back_fill.isna().sum().sum() > 0:
            raise ValueError("NaN values found after back-filling.")

        # Open trade
        open_trade_data = data_back_fill[data_back_fill.index.dayofweek == config['opening_trade_day_of_week']]
        print(open_trade_data)
        entry_price = open_trade_data.groupby(open_trade_data.index.date).first()
        entry_price.sort_index(inplace=True)
        # time of day is removed by pandas as there is only one entry per day
        print(entry_price)

        # Closing trade
        closing_trade_num_days_later = config['closing_trade_num_days_later']
        if closing_trade_num_days_later < 1:
            raise ValueError("closing_trade_num_days_later must be at least 1.")

        closing_trade_offset_dates = open_trade_data.index + pd.DateOffset(days=closing_trade_num_days_later)
        closing_trade_data = data_back_fill[data_back_fill.index.isin(closing_trade_offset_dates)]

        if closing_trade_data.isna().sum().sum() != 0:
            raise ValueError("NaN values found in closing_trade_data after back filling.")
        print(closing_trade_data)

        exit_price = closing_trade_data.groupby(closing_trade_data.index.date).last()
        # set equal to the entry price as this is the reference date
        exit_price.index = entry_price.index
        exit_price.sort_index(inplace=True)
        # time of day is removed by pandas as there is only one entry per day

        if exit_price.isna().sum().sum() != 0:
            raise ValueError("NaN values found in exit_price after grouping.")

        print(exit_price)

        #
        # Calc returns
        #
        single_stock_returns = pd.DataFrame(
            data=exit_price['close'] / entry_price['close'] - 1,
            index=entry_price.index
        )
        single_stock_returns.rename(columns={'close': ticker}, inplace=True)
        single_stock_returns.sort_index(inplace=True)
        print(single_stock_returns)

        if single_stock_returns.isna().sum().sum() != 0:
            raise ValueError("NaN values found in single_stock_returns.")
        print(single_stock_returns.shape)

        return single_stock_returns

    except pd.errors.EmptyDataError:
        print(f"Received empty data for ticker {ticker}")
    except KeyError as e:
        print(f"Missing key: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Error processing returns for {ticker}: {e}")

    return pd.DataFrame()


def get_targets(config):
    all_stock_returns = []

    for ticker in config['tickers']:
        single_stock_returns = calc_returns(ticker, config)
        if not single_stock_returns.empty:
            all_stock_returns.append(single_stock_returns)

    if not all_stock_returns:
            raise ValueError("No valid returns data available.")

    combined_returns = pd.concat(all_stock_returns, axis='columns')
    print(combined_returns)
    print(combined_returns.shape)

    ranked_returns = combined_returns.rank(axis=1)
    print(ranked_returns)
    print(combined_returns.shape)

    # Bucket the columns into 5 buckets for each row
    num_bins = config['target_num_bins']
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    bucketed_returns = ranked_returns.apply(lambda x: pd.qcut(x, num_bins, labels=False), axis=1)
    # bucketed_returns.to_pickle('raw_prices_directory' + 'targets.pkl')
    print(bucketed_returns)
    print(bucketed_returns.shape)

    return bucketed_returns
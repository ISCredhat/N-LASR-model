import pandas as pd

from SMB.config import DATA_RESAMPLE_INTERVAL, DATA_EXCHANGE_START_TIME, DATA_EXCHANGE_END_TIME, \
    TARGETS_CLOSING_TRADE_NUM_DAYS_LATER, TARGETS_OPEN_TRADE_DAY_OF_WEEK, TICKERS, TARGETS_NUM_BINS, LIMIT_DROP_NA_ROWS
from SMB.data import read_prices_file, resample_filter_data
from my_logger import get_logger


logger = get_logger()


def _calc_single_stock_returns(ticker, config):
    try:
        # Ensure config has all required fields
        required_keys = [DATA_RESAMPLE_INTERVAL, DATA_EXCHANGE_START_TIME, DATA_EXCHANGE_END_TIME, TARGETS_OPEN_TRADE_DAY_OF_WEEK,
                         TARGETS_CLOSING_TRADE_NUM_DAYS_LATER]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing '{key}' in configuration.")

        #
        # Get raw data from file
        #
        data = read_prices_file(ticker)
        logger.debug(f"Data shape after reading file for ticker {ticker}: {data.shape}")

        #
        # Resample to 'DATA_RESAMPLE_INTERVAL', remove weekends and times outside exchange hours
        #
        resampled_filtered_data = resample_filter_data(data, config[DATA_RESAMPLE_INTERVAL], config[DATA_EXCHANGE_START_TIME],
                                                       config[DATA_EXCHANGE_END_TIME])
        resampled_filtered_data['date_time'] = resampled_filtered_data.index
        logger.debug(f"Data shape after resampling and filtering for ticker {ticker}: {resampled_filtered_data.shape}")

        #
        # For the OPENING & CLOSING trades we use back-filled prices so we can't trade in the past
        #

        data_back_fill = resampled_filtered_data.bfill()
        logger.debug(f"Data shape after back-filling for ticker {ticker}: {data_back_fill.shape}")

        if data_back_fill.isna().sum().sum() > 0:
            raise ValueError("NaN values found after back-filling.")

        # Open trade
        open_trade_data = data_back_fill[data_back_fill.index.dayofweek == config[TARGETS_OPEN_TRADE_DAY_OF_WEEK]]
        entry_price = open_trade_data.groupby(open_trade_data.index.date).first()
        entry_price.set_index('date_time', inplace=True)
        entry_price.sort_index(inplace=True)
        # time of day is removed by pandas as there is only one entry per day

        logger.debug(f"Entry prices for ticker {ticker}: {entry_price}")

        # Closing trade
        targets_closing_trade_num_days_later = config[TARGETS_CLOSING_TRADE_NUM_DAYS_LATER]
        if targets_closing_trade_num_days_later < 1:
            raise ValueError("targets_closing_trade_num_days_later must be at least 1.")

        closing_trade_offset_dates = open_trade_data.index + pd.DateOffset(days=targets_closing_trade_num_days_later)
        closing_trade_data = data_back_fill[data_back_fill.index.isin(closing_trade_offset_dates)]

        if closing_trade_data.isna().sum().sum() != 0:
            raise ValueError("NaN values found in closing_trade_data after back filling.")

        exit_price = closing_trade_data.groupby(closing_trade_data.index.date).last()
        exit_price.index = entry_price.index
        exit_price.sort_index(inplace=True)

        if exit_price.isna().sum().sum() != 0:
            raise ValueError("NaN values found in exit_price after grouping.")


        logger.debug(f"Exit prices for ticker {ticker}: {exit_price}")
        #
        # Calc returns
        #
        single_stock_returns = pd.DataFrame(
            data=exit_price['close'] / entry_price['close'] - 1,
            index=entry_price.index
        )
        single_stock_returns.rename(columns={'close': ticker}, inplace=True)
        single_stock_returns.sort_index(inplace=True)
        if single_stock_returns.isna().sum().sum() != 0:
            raise ValueError("NaN values found in single_stock_returns.")

        logger.debug(f"Single stock returns for ticker {ticker}: {single_stock_returns}")

        return single_stock_returns

    except pd.errors.EmptyDataError:
        logger.error(f"Received empty data for ticker {ticker}")
    except KeyError as e:
        logger.error(f"Missing key: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except Exception as e:
        logger.error(f"Error processing returns for {ticker}: {e}")
    return pd.DataFrame()


def _calc_all_stock_returns(config):
    all_stock_returns = []

    for ticker in config[TICKERS]:
        single_stock_returns = _calc_single_stock_returns(ticker, config)
        if not single_stock_returns.empty:
            all_stock_returns.append(single_stock_returns)
        else:
            msg = "single_stock_returns is Empty."
            logger.error(msg)
            raise ValueError(msg)

    if not all_stock_returns:
        msg = "No valid returns data available."
        logger.error(msg)
        raise ValueError(msg)

    combined_returns = pd.concat(all_stock_returns, axis='columns')
    logger.debug(f"Combined returns: {combined_returns}")

    return combined_returns


def drop_na_rows_with_limit(name, df, limit):
    df_drop_na = df.dropna()
    num_rows_dropped = df.shape[0] - df_drop_na.shape[0]
    logger.debug(f"Dropped {num_rows_dropped} rows due to missing values.")

    if num_rows_dropped > limit:
        logger.warning(f"'{name}', Dropped more rows than permitted. Dropped {num_rows_dropped} rows due to missing values for limit:'{limit}'.")
        logger.warning(f"")

    return df_drop_na


#
# Calculates the returns for each stock, ranks them by ROW, quantiles the ranks by ROW
#
def calc_targets(config, use_top_bottom_only):
    def _calc_bucketed_returns(returns):
        # method='first' MUST be used to ensure distinctly different rank values
        # We are ranking along the rows so axis='columns'.
        ranked_returns = all_stock_returns_drop_na.rank(axis="columns", method='first', ascending=True,
                                                        numeric_only=True)
        logger.debug(f"Ranked returns: {ranked_returns}")

        # Bucket the columns into 5 buckets for each row
        # We are bucketing along the rows so axis='columns'.
        num_bins = config[TARGETS_NUM_BINS]
        if num_bins < 1:
            msg = "num_bins must be at least 1."
            logger.error(msg)
            raise ValueError(msg)

        bucketed_returns = ranked_returns.apply(lambda x: pd.qcut(x, num_bins, labels=False), axis='columns')
        if use_top_bottom_only:
            bucketed_returns.replace({1: 0, 2: 0, 3: 0}, inplace=True)

        logger.warning(f"Bucketed returns: {bucketed_returns}")
        return bucketed_returns


    def _calc_signed_returns(returns):
        signed_returns = returns.copy()
        signed_returns[signed_returns <= 0] = 0
        signed_returns[signed_returns > 0] = 1
        return signed_returns


    all_stock_returns = _calc_all_stock_returns(config)

    # we must remove rows with na as they are not permitted by 'qcut'
    all_stock_returns_drop_na = drop_na_rows_with_limit('targets', all_stock_returns, config[LIMIT_DROP_NA_ROWS])

    return _calc_bucketed_returns(all_stock_returns_drop_na), _calc_signed_returns(all_stock_returns_drop_na), all_stock_returns_drop_na

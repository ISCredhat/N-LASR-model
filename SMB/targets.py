import pandas as pd
from SMB.data import read_prices_file, resample_filter_data
import logging
logger = logging.getLogger(__name__)


def calc_returns(ticker, config):
    try:
        # Ensure config has all required fields
        required_keys = ['resample_interval', 'exchange_start_time', 'exchange_end_time', 'opening_trade_day_of_week',
                         'closing_trade_num_days_later']
        for key in required_keys:
            if key not in config:
                msg = f"Missing '{key}' in configuration."
                logger.error(msg)
                raise ValueError(msg)

        #
        # Get raw data from file
        #
        data = read_prices_file(ticker)
        logger.debug(f"Data shape after reading file for ticker {ticker}: {data.shape}")

        #
        # Resample to 'resample_interval', remove weekends and times outside exchange hours
        #
        resampled_filtered_data = resample_filter_data(data, config['resample_interval'], config['exchange_start_time'],
                                                       config['exchange_end_time'])
        logger.debug(f"Data shape after resampling and filtering for ticker {ticker}: {resampled_filtered_data.shape}")

        #
        # For the OPENING & CLOSING trades we use back-filled prices so we can't trade in the past
        #

        # it may be possible to do a forward and backwards fill here
        # data_fwd_fill = resampled_filtered_data.ffill()
        # print(data_fwd_fill.iloc[:100, :])
        # print(data_fwd_fill.isna().sum())
        # print(data_fwd_fill.shape)

        data_back_fill = resampled_filtered_data.bfill()
        logger.debug(f"Data shape after back-filling for ticker {ticker}: {data_back_fill.shape}")

        if data_back_fill.isna().sum().sum() > 0:
            msg = "NaN values found after back-filling."
            logger.error(msg)
            raise ValueError(msg)

        # Open trade
        open_trade_data = data_back_fill[data_back_fill.index.dayofweek == config['opening_trade_day_of_week']]
        entry_price = open_trade_data.groupby(open_trade_data.index.date).first()
        entry_price.sort_index(inplace=True)
        # time of day is removed by pandas as there is only one entry per day

        logger.debug(f"Entry prices for ticker {ticker}: {entry_price}")

        # Closing trade
        closing_trade_num_days_later = config['closing_trade_num_days_later']
        if closing_trade_num_days_later < 1:
            msg = "closing_trade_num_days_later must be at least 1."
            logger.error(msg)
            raise ValueError(msg)

        closing_trade_offset_dates = open_trade_data.index + pd.DateOffset(days=closing_trade_num_days_later)
        closing_trade_data = data_back_fill[data_back_fill.index.isin(closing_trade_offset_dates)]

        if closing_trade_data.isna().sum().sum() != 0:
            msg = "NaN values found in closing_trade_data after back filling."
            logger.error(msg)
            raise ValueError(msg)

        exit_price = closing_trade_data.groupby(closing_trade_data.index.date).last()
        exit_price.index = entry_price.index
        exit_price.sort_index(inplace=True)

        if exit_price.isna().sum().sum() != 0:
            msg = "NaN values found in exit_price after grouping."
            logger.error(msg)
            raise ValueError(msg)


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
            msg = "NaN values found in single_stock_returns."
            logger.error(msg)
            raise ValueError(msg)

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

#
# Calculates the returns for each stock and
#
def get_targets(config):
    all_stock_returns = []

    for ticker in config['tickers']:
        single_stock_returns = calc_returns(ticker, config)
        if not single_stock_returns.empty:
            all_stock_returns.append(single_stock_returns)

    if not all_stock_returns:
        msg = "No valid returns data available."
        logger.error(msg)
        raise ValueError(msg)

    combined_returns = pd.concat(all_stock_returns, axis='columns')
    logger.debug(f"Combined returns: {combined_returns}")

    ranked_returns = combined_returns.rank(axis=1)
    logger.debug(f"Ranked returns: {ranked_returns}")

    # Bucket the columns into 5 buckets for each row
    num_bins = config['target_num_bins']
    if num_bins < 1:
        msg = "num_bins must be at least 1."
        logger.error(msg)
        raise ValueError(msg)
    bucketed_returns = ranked_returns.apply(lambda x: pd.qcut(x, num_bins, labels=False), axis=1)
    logger.debug(f"Bucketed returns: {bucketed_returns}")

    return bucketed_returns


def _reshape(df):
    # Melt the DataFrame
    reshaped_df = df.melt(var_name='variable', value_name='value')

    # Extracting only the values column
    reshaped_df = reshaped_df[['value']]

    logging.info(f"original shape:{df.shape}")
    logging.info(f"new shape:{reshaped_df.shape}")

    return reshaped_df

#
# We need to reshape the targets to be a single column of all stocks stacked
#
# noinspection PyPep8Naming
def reshape_X_y(X, y):
    reshaped_X = _reshape(X)
    reshaped_y = _reshape(y)

    return reshaped_X, reshaped_y
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split

from my_logger import get_logger


from SMB.config import FEATURES_NUM_BINS, LIMIT_DROP_NA_ROWS, DATA_RESAMPLE_INTERVAL, \
    TARGETS_CLOSING_TRADE_NUM_DAYS_LATER, DATA_EXCHANGE_START_TIME, DATA_EXCHANGE_END_TIME, \
    TARGETS_OPEN_TRADE_DAY_OF_WEEK, FEATURES_INDICATOR_ROLLING_WINDOW, TICKERS, FEATURES_TARGETS_INDEX_DIFFERENCE_LIMIT, \
    read_config, TRAIN_TEST_SPLIT, TRAIN_TEST_RANDOM_STATE, TRAIN_TEST_SHUFFLE
from SMB.data import read_prices_file, resample_filter_data
from SMB.targets import drop_na_rows_with_limit

logger = get_logger()

def _calc_single_stick_indicators(p, rolling_window):
    def _subtract_and_normalize(data_series, px_open):
        return (data_series - px_open) / px_open

    def _calc_diff_from_mean(indicator, rolling_window):
        return indicator - indicator.rolling(rolling_window).mean()

    indicators = pd.DataFrame(index=p.index)

    if p.isna().sum().sum() != 0:
        msg = "NaN values found in price data."
        raise ValueError(msg)

    if pd.Series(p.close == 0).any():
        msg = "There are values in 'p.close' equal to zero."
        logger.error(msg)
        raise ValueError(msg)

    data = {
        'adx': ta.adx(p.high, p.low, p.close)['ADX_14'],  # returns pd.DataFrame: adx, dmp, dmn
        'alma': _subtract_and_normalize(ta.alma(p.close), p.open),
        'ao': _subtract_and_normalize(ta.ao(p.high, p.low), p.open),
        'atr': _subtract_and_normalize(ta.atr(p.high, p.low, p.close), p.open),
        'bias': _subtract_and_normalize(ta.bias(p.close), p.open),
        'bop': ta.bop(p.open, p.high, p.low, p.close),
        # too many NAN - need to fix
        # 'cci': ta.cci(p.high, p.low, p.close),
        'cfo': _subtract_and_normalize(ta.cfo(p.close), p.open),
        'chop': _subtract_and_normalize(ta.chop(p.high, p.low, p.close), p.open),
        'cmf': ta.cmf(p.high, p.low, p.close, p.volume),
        'cmo': ta.cmo(p.close),
        # too many NAN - need to fix
        # 'corr': p.close.rolling(window=rolling_window).corr(p.close.shift(1)),
        
        #
        # Throws an error so commented out
        #
        # /Users/stephanie/src/N-LASR-model/.venv/lib/python3.12/site-packages/pandas_ta/overlap/linreg.py:53: RuntimeWarning: invalid value encountered in scalar divide
        #   return rn / rd
        # /Users/stephanie/src/N-LASR-model/.venv/lib/python3.12/site-packages/pandas_ta/overlap/linreg.py:52: RuntimeWarning: invalid value encountered in scalar power
        #   rd = (divisor * (length * y2_sum - y_sum * y_sum)) ** 0.5
        # cti = (ta.cti(p.close) - p.open) / p.open
        # indicators['cti'] = calc_mean(cti, rolling_window)

        # this may be wrong
        'dmp_14': _subtract_and_normalize(ta.dm(p.high, p.low)['DMP_14'], p.open),
        'dema': _subtract_and_normalize(ta.dema(p.close), p.open),
        'dpo': ta.dpo(p.close),
        'efi': _subtract_and_normalize(ta.efi(p.close, p.volume), p.open),

        # modified compared to Isaac as I specify the BULLP_13
        'eri_bullp_13': _subtract_and_normalize(ta.eri(p.high, p.low, p.close).BULLP_13, p.open),
        'eri_bear': _subtract_and_normalize(ta.eri(p.high, p.low, p.close).BEARP_13, p.open),
        'ema': _subtract_and_normalize(ta.ema(p.close), p.open),
        # FIXME I'm unsure why this seems to return all NaN but could be a single stock breaking it???
        # 'eom': ta.eom(p.high, p.low, p.close, p.volume),
        'fisher_9_1': ta.fisher(p.high, p.low)['FISHERT_9_1'].astype('float64'),
        'fwma': _subtract_and_normalize(ta.fwma(p.close), p.open),
        'hl2': _subtract_and_normalize(ta.hl2(p.high, p.low), p.open),
        'hilo_13_21': _subtract_and_normalize(ta.hilo(p.high, p.low, p.close)['HILO_13_21'], p.open),
        'hma': _subtract_and_normalize(ta.hma(p.close), p.open),
        'hwma': _subtract_and_normalize(ta.hwma(p.close), p.open),
        'inertia': _subtract_and_normalize(ta.inertia(p.close), p.open),
        'kama': ta.kama(p.close),

        # indicators['kc_upper'] = kc_upper - kc_upper.rolling(rolling_window).mean()
        # pd.DataFrame: lower, basis, upper columns.
        # this is the wrong way to use this - should be close wrt the upper or lower
        'kc_upper_KCUe_20_2': ta.kc(p.high, p.low, p.close).KCUe_20_2,

        # the function returns pd.DataFrame: kst and kst_signal columns so .kst MAY NOT be correct?
        'kst_KST_10_15_20_30_10_10_10_15': ta.kst(p.close)['KST_10_15_20_30_10_10_10_15'],
        'kurtosis': _subtract_and_normalize(ta.kurtosis(p.close), p.open),
        'linreg': _subtract_and_normalize(ta.linreg(p.close), p.open),
        'macd_12_26_9': _subtract_and_normalize(ta.macd(p.close)['MACD_12_26_9'], p.open),
        'median': _subtract_and_normalize(ta.median(p.close), p.open),
        'midpoint': _subtract_and_normalize(ta.midpoint(p.close), p.open),
        'midprice': _subtract_and_normalize(ta.midprice(p.high, p.low), p.open),
        'mom': _subtract_and_normalize(ta.mom(p.close), p.open),

        # Isaac has nvi = nvi.infer_objects()
        'nvi': _subtract_and_normalize(ta.nvi(p.close, p.volume), p.open),
        'obv': _subtract_and_normalize(ta.obv(p.close, p.volume), p.open),
        'percent_return': ta.percent_return(p.close),
        'ppo_12_26_9': _subtract_and_normalize(ta.ppo(p.close)['PPO_12_26_9'], p.open),
        'pvi': _subtract_and_normalize(ta.pvi(p.close, p.volume), p.open),
        'pvo_12_26_9': _subtract_and_normalize(ta.pvo(p.volume)['PVO_12_26_9'], p.open),
        'pvt': _subtract_and_normalize(ta.pvt(p.close, p.volume), p.open),
        'pwma': _subtract_and_normalize(ta.pwma(p.close), p.open),
        'mad': _subtract_and_normalize(ta.mad(p.close), p.open),
        'roc': _subtract_and_normalize(ta.roc(p.close), p.open),
        'rsi': ta.rsi(p.close),
        'sinwma': _subtract_and_normalize(ta.sinwma(p.close), p.open),
        'skew': _subtract_and_normalize(ta.skew(p.close), p.open),
        'slope': _subtract_and_normalize(ta.slope(p.close), p.open),
        'sma': _subtract_and_normalize(ta.sma(p.close), p.open),
        'stddev': _subtract_and_normalize(ta.stdev(p.close), p.open),
        'supertrend_7_3.0': _subtract_and_normalize(ta.supertrend(p.high, p.low, p.close)['SUPERT_7_3.0'], p.open),
        't3': _subtract_and_normalize(ta.t3(p.close), p.open),
        'tema': _subtract_and_normalize(ta.t3(p.close), p.open),
        'trima': _subtract_and_normalize(ta.trima(p.close), p.open),
        'trix_30_9': _subtract_and_normalize(ta.trix(p.close)['TRIX_30_9'], p.open),
        'true_range': _subtract_and_normalize(ta.true_range(p.high, p.low, p.close), p.open),
        'uo': ta.uo(p.high, p.low, p.close),
        'variance': _subtract_and_normalize(ta.variance(p.close), p.open),

        # indicators['vortex'] = vortex - vortex.rolling(rolling_window).mean()
        # this is the wrong way to use this - should be close wrt the upper or lower?
        # https: // stockcharts.com / school / doku.php?id = chart_school:technical_indicators: vortex_indicator
        'vortex_vtxp_14': ta.vortex(p.high, p.low, p.close).VTXP_14,
        'wma': _subtract_and_normalize(ta.wma(p.close), p.open),
        'willr': ta.willr(p.high, p.low, p.close)
    }

    for key, value in data.items():
        try:
            indicators[key] = _calc_diff_from_mean(value, rolling_window)
        except Exception as e:
            logger.error(f"Error processing returns for key:{key}: {e}")

    return indicators


def _calc_single_stock_indicator(ticker, config):
    try:
        # Ensure config has all required fields
        required_keys = [DATA_RESAMPLE_INTERVAL, DATA_EXCHANGE_START_TIME, DATA_EXCHANGE_END_TIME, TARGETS_OPEN_TRADE_DAY_OF_WEEK,
                         TARGETS_CLOSING_TRADE_NUM_DAYS_LATER, FEATURES_INDICATOR_ROLLING_WINDOW]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing '{key}' in configuration.")

        #
        # Get raw data from file
        #
        logger.debug(f"getting data for:{ticker}")
        data = read_prices_file(ticker)
        logger.debug(data.head())
        logger.debug(data.columns)
        logger.debug(f"Data shape after reading file for ticker {ticker}: {data.shape}")

        #
        # Resample to 'DATA_RESAMPLE_INTERVAL', remove weekends and times outside exchange hours
        #
        resampled_filtered_data = resample_filter_data(data, config[DATA_RESAMPLE_INTERVAL], config[DATA_EXCHANGE_START_TIME],
                                                       config[DATA_EXCHANGE_END_TIME])
        logger.debug(f"Data shape after resampling and filtering for ticker {ticker}: {resampled_filtered_data.shape}")

        #
        # For missing data we will fill FORWARD so that we don't look into the future.
        #
        num_na_before_fill = resampled_filtered_data.isna().sum().sum()
        data_fwd_fill = resampled_filtered_data.ffill()
        num_na_after_fill = data_fwd_fill.isna().sum().sum()
        if num_na_after_fill > 0:
            raise ValueError("NaN values found after back-filling.")
        logger.debug(f"For '{ticker}' num data forward filled:'{num_na_before_fill-num_na_after_fill}'")

        indicators = _calc_single_stick_indicators(data_fwd_fill, config[FEATURES_INDICATOR_ROLLING_WINDOW])
        if indicators.empty:
            raise ValueError(f"'{ticker}': indicators is empty.")
        # we expect NaN due to ta indicators only returning valid values after a min amount of data
        # if indicators.isna().sum().sum() != 0:
        #     raise ValueError("NaN values found in indicators.")

        return indicators

    except pd.errors.EmptyDataError:
        logger.error(f"Received empty data for ticker {ticker}")
    except KeyError as e:
        logger.error(f"Missing key: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except Exception as e:
        logger.error(f"Error processing returns for {ticker}: {e}")
    return pd.DataFrame()

def _calc_all_stock_indicators(config):
    all_stock_indicators_dict = {}

    for ticker in config[TICKERS]:
        indicators = _calc_single_stock_indicator(ticker, config)
        if not indicators.empty:
            all_stock_indicators_dict[ticker] = indicators
        else:
            msg = "calc_daily_indicators is Empty."
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Calculated indicators for:'{ticker}'.")

    if not all_stock_indicators_dict:
        msg = "No valid indicators available."
        logger.error(msg)
        raise ValueError(msg)

    combined_all_stock_indicators = pd.concat(all_stock_indicators_dict, axis='columns')

    return combined_all_stock_indicators


#
# Calculates the features for each stock and stores in a multi-index dataframe
#
def calc_features(config, dates_index):
    combined_all_stock_indicators = _calc_all_stock_indicators(config)

    # we only need the indicators for the date_times of the targets
    filtered_combined_all_stock_indicators = combined_all_stock_indicators.loc[dates_index]

    num_bins = config[FEATURES_NUM_BINS]
    all_stock_bucketed_indicator_values = {}
    for indicator_name in filtered_combined_all_stock_indicators.columns.get_level_values(1).unique():
        indicator_values = filtered_combined_all_stock_indicators.xs(indicator_name, level=1, axis='columns')

        # we must remove rows with na as they are not permitted by 'qcut'
        indicator_values_drop_na = drop_na_rows_with_limit(indicator_name, indicator_values, config[LIMIT_DROP_NA_ROWS])

        # method='first' MUST be used to ensure distinctly different rank values
        # We are ranking along the rows so axis='columns'.
        # TODO explore whether to rank by row or column
        ranked_indicator_values = indicator_values_drop_na.rank(axis="columns", method='first', ascending=True)
        logger.info(f"'{indicator_name}', ranked_indicator_values'{ranked_indicator_values.shape}'.")
        logger.debug(ranked_indicator_values.tail())

        # Bucket the columns into 5 buckets for each row
        # We are bucketing along the rows so axis='columns'.
        if num_bins < 1:
            msg = "num_bins must be at least 1."
            raise ValueError(msg)
        # TODO explore whether to rank by row or column
        bucketed_indicator_values = ranked_indicator_values.apply(lambda x: pd.qcut(x, num_bins, labels=False), axis='columns')
        logger.debug(f"'{indicator_name}', Bucketed returns: {bucketed_indicator_values}")

        if not bucketed_indicator_values.empty:
            all_stock_bucketed_indicator_values[indicator_name] = bucketed_indicator_values
            logger.info(f"'{indicator_name}', length bucketed_indicator_values:'{len(bucketed_indicator_values.keys())}'.")
        else:
            msg = f"'{indicator_name}', bucketed_indicator_values is Empty."
            logger.error(msg)
            raise ValueError(msg)

    if not all_stock_bucketed_indicator_values:
        msg = "No valid targets available."
        logger.error(msg)
        raise ValueError(msg)

    return pd.concat(all_stock_bucketed_indicator_values, axis='columns')

#
# align the indices of features and targets to take account of the data lost
# to indicator warm up
#
def align_targets_and_features(config, features, targets):
    if features.index.equals(targets.index):
        logger.info("Indices match!")
    else:
        logger.info(f"Indices do not match! num of rows, features:'{features.shape[0]}', targets:'{targets.shape[0]}'")

        diff_features_index = features.index.difference(targets.index)
        diff_targets_index = targets.index.difference(features.index)

        logger.info(f"\nRows in features but not in targets:\n{diff_features_index}")
        logger.info(f"\nRows in targets but not in features:\n{diff_targets_index}")

        different_features_rows = features.loc[diff_features_index]
        different_targets_rows = targets.loc[diff_targets_index]

        logger.info(f"\nDifferent rows in features:\n{different_features_rows}")
        logger.info(f"\nDifferent rows in targets:\n{different_targets_rows}")

        if abs(features.shape[0] - targets.shape[0]) > config[FEATURES_TARGETS_INDEX_DIFFERENCE_LIMIT]:
            logger.warning('WARNING indices TOO different')

        # FIXME
        # if diff_features_index.shape[0] > 0:
        #     msg = "diff_features_index index is NOT zero."
        #     logger.error(msg)
        #     raise ValueError(msg)

        return features, targets.loc[features.index]


def reshape_X(X):
    # stack the rows -> date, stock ticker row index and indicator column names
    # swap row index 0 and 1 -> stock ticker, date
    # sort the rows by stock ticker and then date
    # sort the indicator columns
    return (X.stack(1, future_stack=True)
        .swaplevel(1, 0)
        .sort_index(level=[0, 1], axis='columns')
        .sort_index(level=[0, 1], axis='rows')
    )


def reshape_y(y):
    # stack the rows -> date, stock ticker row index and indicator column names
    # swap row index 0 and 1 -> stock ticker, date
    # sort the rows by stock ticker and then date
    return (y.stack(0, future_stack=True)
        .swaplevel(0, 1)
        .sort_index(level=[0, 1], axis='rows')
    )


#
# We need to reshape the targets to be a single column of all stocks stacked
#
# noinspection PyPep8Naming
def reshape_X_y(X, y):
    # stack the rows -> date, stock ticker row index and indicator column names
    # swap row index 0 and 1 -> stock ticker, date
    # sort the rows by stock ticker and then date
    # sort the indicator columns
    X_reshaped = reshape_X(X)

    # stack the rows -> date, stock ticker row index and indicator column names
    # swap row index 0 and 1 -> stock ticker, date
    # sort the rows by stock ticker and then date
    y_reshaped = reshape_y(y)

    logger.debug(f"original shape:{X.shape}, new shape:{X_reshaped.shape}")
    logger.debug('.') # Added to prevent a bug that duplicate the line
    logger.debug(f"original shape:{y.shape}, new shape:{y_reshaped.shape}")

    return X_reshaped, y_reshaped


def generate_training_test_data(config_name, X, y):
    #
    # generate the train test split_sections
    #
    config = read_config(config_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config[TRAIN_TEST_SPLIT],
        random_state=config[TRAIN_TEST_RANDOM_STATE],
        shuffle=config[TRAIN_TEST_SHUFFLE]
    )

    #
    # reshape the data.
    # it is important to do this AFTER the splitting to ensure all rows of a stock are split over the train and test data.
    #
    X_train_stacked, y_train_stacked = reshape_X_y(X_train, y_train)
    logger.info(f"Shapes: X:'{X.shape}', X_train:'{X_train.shape}', X_train_stacked:'{X_train_stacked.shape}'.")
    logger.info(f"Shapes: y:'{y.shape}', y_train:'{y_train.shape}', y_train_stacked:'{y_train_stacked.shape}'.")

    X_test_stacked, y_test_stacked = reshape_X_y(X_test, y_test)
    logger.info(f"Shapes: X:'{X.shape}', X_train:'{X_test.shape}', X_train_stacked:'{X_test_stacked.shape}'.")
    logger.info(f"Shapes: y:'{y.shape}', y_train:'{y_test.shape}', y_train_stacked:'{y_test_stacked.shape}'.")

    return X_train_stacked, y_train_stacked, X_test_stacked, y_test_stacked
# from datetime import datetime
#
# import numpy as np
#
# from SMB.load_data import get_data
# import pandas as pd
#
# # import os
# # os.chdir('SMB')
# import matplotlib.pyplot as plt
#
# def calc_stock_returns(prices):
#     prices.set_index(pd.to_datetime(prices['date']), inplace=True)
#
#     mondays = prices[prices.index.weekday == 0]  # Mondays
#     mondays_after_9 = mondays[(mondays.index.hour >= 9)]
#     monday_prices = mondays_after_9.groupby(mondays_after_9.index.date).first()
#     print(monday_prices)
#
#     fridays = prices[prices.index.weekday == 4]  # Fridays
#     fridays_before_22 = fridays[(fridays.index.hour < 22)]
#     friday_prices = fridays_before_22.groupby(fridays_before_22.index.date).last()
#     print(friday_prices)
#
#     # weekly_first_monday = prices.resample('W-MON')['close'].first()
#     # weekly_last_friday = prices.resample('W-FRI')['close'].last()
#
#     index_adjustment = pd.to_datetime(friday_prices.index) - pd.to_timedelta(4, unit='d')
#     friday_prices.index = index_adjustment
#
#     df = pd.DataFrame({
#         'return': (friday_prices['close'] / monday_prices['close'] - 1),
#         'monday_datetime': monday_prices['date'],
#         'monday_close': monday_prices['close'],
#         'friday_datetime': friday_prices['date'],
#         'friday_close': friday_prices['close'],
#     })
#     pd.set_option("display.max_columns", None)
#     print(df)
#
#     monday_datetime = pd.DataFrame({
#         'monday_datetime': weekly_first_monday.index.strftime('%Y-%m-%d %H:%M:%S')})
#     friday_datetime = pd.DataFrame({
#         'friday_datetime': weekly_last_friday.index.strftime('%Y-%m-%d %H:%M:%S')})
#     stock_returns = pd.concat([weekly_return, monday_datetime, friday_datetime], axis=1)
#     # print(df2)
#     # weekly_return.plot(); plt.show()
#
#     return returns_df
#
# def calc_all_stock_targets(all_stock_prices):
#     all_stock_targets_dict = {}
#     for ticker in all_stock_prices.columns.get_level_values(0).unique():
#         targets = calc_stock_targets(all_stock_prices[ticker])
#         all_stock_targets_dict[ticker] = targets
#         print('Calculated targets for:', ticker)
#
#     combined_all_stock_targets = pd.concat(all_stock_targets_dict, axis=1)
#
#     return combined_all_stock_targets
#
# all_stock_targets = calc_all_stock_targets(get_data())
# all_stock_targets.to_pickle('../data/all_stock_targets.pkl')
#
# prices = get_data()['AIG']
#
# daily_df = prices.resample('h').first()
# all_days = pd.date_range(start=prices.index.min().date(), end=prices.index.max().date(), freq='D')
# rescaled_df = daily_df.reindex(all_days)
# print(rescaled_df)
#
# mondays = prices[prices.index.weekday == 0]
# mondays_after_9 = mondays[(mondays.index.hour >= 9)]
# monday_prices = mondays_after_9.groupby(mondays_after_9.index.date).first()
# monday_prices.set_index(pd.to_datetime(monday_prices['date']), inplace=True)
# print(monday_prices)
#
# fridays = prices[prices.index.weekday == 4]
# fridays_before_22 = fridays[(fridays.index.hour <= 22)]
# friday_prices = fridays_before_22.groupby(fridays_before_22.index.date).last()
# friday_prices.set_index(pd.to_datetime(friday_prices['date']), inplace=True)
# print(friday_prices)
#
# df = pd.DataFrame(rescaled_df)
# pd.concat([df, monday_prices, friday_prices], axis=1)
#
#
#
# print(returns_df)
#
import pandas as pd
from SMB.load_data import read_data, resample_filter_data


def calc_returns(ticker, config):
    #
    # get raw data from file
    #
    data = read_data(ticker)
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
    print(single_stock_returns.shape)

    if single_stock_returns.isna().sum().sum() != 0:
        raise ValueError("NaN values found in single_stock_returns.")
    print(single_stock_returns.shape)
    print(single_stock_returns)

    return single_stock_returns

def get_targets(config):
    all_stock_returns = []

    for ticker in config['tickers']:
        single_stock_returns = calc_returns(ticker, config)
        all_stock_returns.append(single_stock_returns)

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
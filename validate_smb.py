import pandas as pd

total_model_trading = pd.read_csv('./data/total_model_trading_dataframe.csv')
# Index(['Original Index', 'Values', 'Ticker', 'Forward Return',
#        'Monday Datetime', 'Monday First Hour Close', 'Friday Last Hour Open'],
#       dtype='object')

total_trading = pd.read_csv('./data/total_trading_dataframe.csv')
# Index(['Original Index', 'Values', 'Ticker', 'Forward Return',
#        'Monday Datetime', 'Monday First Hour Close', 'Friday Last Hour Open'],
#       dtype='object')

pd.set_option("display.max_columns", None)
print(total_model_trading.iloc[1:3,:])

print(total_model_trading['Forward Return'].mean())

import matplotlib.pyplot as plt
total_model_trading['Forward Return'].plot()
plt.show()

import quantstats_lumi as qs
qs.extend_pandas()

idx = pd.to_datetime(total_model_trading['Monday Datetime'])
total_model_trading.set_index(idx, inplace=True)
total_model_trading.tz_localize('GMT')

ret = total_model_trading['Forward Return'] / 100
mean_ret_per_week = ret.groupby(level=0).mean()
qs.reports.html(mean_ret_per_week, output='tearSheet.html')
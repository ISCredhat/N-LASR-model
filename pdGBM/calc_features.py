import pandas_ta as ta
# danielverissimo commented on Jun 20 •
# Until the next release, on momentum/squeeze_pro.py change:
# "from numpy import NaN as npNaN" to "import numpy as np"

# import os
# os.getcwd()
# os.chdir('pdGBM')

import pandas as pd
from pdGBM.get_config import base_path

px = pd.read_pickle(base_path + 'dataFrames/prices')

# VWAP requires the DataFrame index to be a DatetimeIndex.
# Replace "datetime" with the appropriate column from your DataFrame
# df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

# Calculate Returns and append to the df DataFrame
px.ta.log_return(cumulative=True, append=True)
px.ta.percent_return(cumulative=True, append=True)

# New Columns with results
print(px.columns)

# Take a peek
print(px.tail())

# we use the following features at k of [0.5 0.75 1:10]
# rsi 14*k with range 0-100 where 50 is the no gain/loss value
# macd with fast and slow parms being round(12*k), round(26*k)
# bbands with bollb = (oneDay(:,2) - lowr) ./ (uppr - lowr) from
# [mid, uppr, lowr] = bollinger(oneDay(:,2), 20*k);
# label the features

features = pd.DataFrame()
for k in [0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    length = round(14 * k)
    feature_name = 'RSI_' + str(length)
    features[feature_name] = px.ta.rsi(length=length, scalar=100, offset=0)

for k in [0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    fast = round(12 * k)
    slow = round(26 * k)
    signal = 9
    results_key = 'MACD_' + str(fast) + '_' + str(slow) + '_' + str(signal)
    print(results_key)
    feature_name = results_key
    features[feature_name] = px.ta.macd(fast=fast, slow=slow)[results_key]

for k in [0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    length = round(20 * k)
    mode = 'sma'
    std = 2.0
    results_key = 'BBP_' + str(length) + '_' + "{:.1f}".format(std)
    print(results_key)
    feature_name = results_key + '_' + mode
    features[feature_name] = px.ta.bbands(length=length, std=2, mamode=mode, ddof=0)[results_key]


print(features)
print(features.columns)

features.to_pickle(base_path + 'dataFrames/features')
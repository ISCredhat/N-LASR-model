from pdGBM.get_config import base_path
import scipy.io as sio
import pandas as pd

mat_contents = sio.loadmat(base_path + 'mdata/AAPL.mat')
print(mat_contents)
mdata = mat_contents['data']

# Assuming 'data' is your variable with mat contents
# Let's assume 'data' is a numpy array for this example
# data = ...

# Create a DataFrame from 'data'
px = pd.DataFrame(mdata)
px.columns = ['datetime', 'close', 'volume', 'num_trades']

from datetime import datetime

# Matlab considers the origin January 0, 0000 and outputs the date as the number of days since then.
# This creates a bit of an issue because that's not a real date and well outside of the datetime64[ns] bounds.
# With a simple subtraction relative to the POSIX origin (1970-01-01) you can then use the vectorized
# pd.to_datetime conversion.offset = datetime(0, 0, 1).toordinal() + 366  #719529
offset = datetime(1970, 1, 1).toordinal() + 366  #719529
idx = pd.to_datetime(px['datetime'] - offset, unit='D', origin='unix')
px.set_index(idx, inplace=True)

# Output the resulting DataFrame
print(px)

px.to_pickle(base_path + 'dataFrames/prices')
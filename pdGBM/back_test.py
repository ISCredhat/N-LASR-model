# Install the latest version from GitHub to incorporate recent fixes:
# pip install git+https://github.com/kernc/backtesting.py.git
# The PyPI version (0.3.3) may not include the necessary fix
#  also see https://github.com/kernc/backtesting.py/issues/1158#issuecomment-2278623466

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
from pdGBM.get_config import base_path

df = pd.read_pickle(base_path + 'dataFrames/df')
df['Close'] = df['close']
df['Open'] = df['close']
df['High'] = df['close']
df['Low'] = df['close']
df.drop(['datetime', 'close', 'volume', 'num_trades'], axis='columns')

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 60)
        self.ma2 = self.I(SMA, price, 300)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(df, SmaCross, commission=.0010, exclusive_orders=True)
stats = bt.run()
print(stats)
bt.plot()
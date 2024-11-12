from pdGBM.get_config import base_path
import pandas as pd
import matplotlib.pyplot as plt

px = pd.read_pickle(base_path + 'dataFrames/prices')

# we predict the percentage price change at horizons of
# [10 20 30 60 120 300] data steps ahead
targets = pd.DataFrame()
for h in [10, 20, 30, 60, 120, 300]:
    target_name = str(h)
    targets[target_name] = px['close'].pct_change(periods=h)

print(targets)
print(targets.columns)
targets.plot()
plt.show()

pd.set_option("display.max_columns", None)
print(targets.describe())

targets.to_pickle(base_path + 'dataFrames/targets')
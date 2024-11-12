import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import pandas as pds

fileName = 'spy100k'
x = np.loadtxt('../IB4m/' + fileName + '_x.csv', delimiter=",")
y = np.loadtxt('../IB4m/' + fileName + '_y.csv', delimiter=",")

dfX = pds.DataFrame(x).iloc[:5000, 0:15]
dfY = pds.DataFrame(y).iloc[:5000, 3]
df = pds.concat([dfY, dfX], axis=1, ignore_index=True)
scatter_matrix(df, alpha=0.1, figsize=(10, 9), marker='.', diagonal='kde')

plt.show()
plt.savefig(fileName + '-y1-x1-10.png')
input("Press Enter to continue...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import ADF_Cal, cal_rolling_mean_var, auto_correlation
from statsmodels.tsa.seasonal import STL

df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv',
                 index_col=0)

df.index = pd.date_range(start='1981-01-01',
                         periods=len(df),
                         freq='D')

print(df)

plt.figure(figsize=(12, 10))
df.plot()
plt.grid()
plt.show()

auto_correlation(df.iloc[:, 0], 50, 'Autocorrelation temperature data')
plt.show()

auto_correlation(df.iloc[:, 0], 100, 'Autocorrelation temperature data')
plt.show()

ADF_Cal(df.iloc[:, 0])

cal_rolling_mean_var(df)

#STL attempt

STL = STL(df)
res = STL.fit()
fig = res.plot()
plt.show()


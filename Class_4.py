#Holt's Winter Method
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv', index_col='Month', parse_dates=True)

print(df.head())

y = df["#Passengers"]

#Train-test split
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

#SES Method
holtt = ets.ExponentialSmoothing(yt,trend=None, damped= False, seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values, np.ndarray.flatten(holtf.values))).mean()
print(MSE)

fig, ax = plt.subplots()
ax.plot(yt,label="Train data")
ax.plot(yf,label="Test data")
ax.plot(holtf,label="Simple exponential smoothing")
plt.legend(loc="upper left")
plt.xlabel("time(monthly)")
plt.ylabel("#of passengers thousands")
plt.show()

#Holt-linear Method
holtt = ets.ExponentialSmoothing(yt,trend='mul', damped= True, seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values, np.ndarray.flatten(holtf.values))).mean()
print(MSE)

fig, ax = plt.subplots()
ax.plot(yt,label="Train data")
ax.plot(yf,label="Test data")
ax.plot(holtf,label="Simple exponential smoothing")
plt.legend(loc="upper left")
plt.xlabel("time(monthly)")
plt.ylabel("#of passengers thousands")
plt.show()

#Holt-winter Method
holtt = ets.ExponentialSmoothing(yt,trend='mul', damped= True, seasonal='mul').fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values, np.ndarray.flatten(holtf.values))).mean()
print(MSE)

fig, ax = plt.subplots()
ax.plot(yt,label="Train data")
ax.plot(yf,label="Test data")
ax.plot(holtf,label="Simple exponential smoothing")
plt.legend(loc="upper left")
plt.xlabel("time(monthly)")
plt.ylabel("#of passengers thousands")
plt.show()



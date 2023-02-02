from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

stocks = ['AAPL','ORCL','TSLA','IBM', 'YELP','MSFT','FRD']
sd = '2000-01-01'
ed = '2023-02-02'
df = data.get_data_yahoo(stocks[0], start = sd, end = ed)

print(df.tail().to_string())

#plotting - viz
col = df.columns
df[col[:-1]].plot()
print(df.describe().to_string())
plt.show()
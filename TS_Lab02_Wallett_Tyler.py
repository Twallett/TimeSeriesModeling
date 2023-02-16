import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Question 2:

def white_noise(random_seed):
    start_date = '01/01/2000'
    end_date = '31/12/2000'
    date = pd.date_range(start_date, end_date, periods=1000)
    np.random.seed(random_seed)
    x = np.random.normal(0, 1, size=1000).round(2)
    df = pd.DataFrame(x, index=date)
    plt.hist(df)
    plt.xlabel('White noise standard deviations')
    plt.ylabel('Frequency')
    plt.title('Histogram of White Noise')
    plt.show()
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('White Noise')
    plt.title('White Noise Time-series')
    plt.show()
    print(f"Sample mean: {x.mean()}")
    print(f"Sample std: {x.std()}")
    return df[0]


wn = white_noise(6313)

# Question 3

q1 = [3, 9, 27, 81, 243]

def auto_correlation(T, tau, title):
    # Y-BAR
    y_bar = np.average(T)

    # DENOMINATOR
    x = []
    for t in range(len(T)):
        x.append(np.sum((T[t] - y_bar) ** 2))
        denom = sum(x)

    # NUMERATOR
    numerator_sum = []
    count = 0
    for i in range(tau + 1):
        for j in range(i, len(T)):
            numerator_sum.append((T[j] - y_bar) * (T[j - count] - y_bar))
        count += 1

    numerator = []
    for i in range(len(T), len(T) - tau - 1, -1):
        numerator.append(sum(numerator_sum[:i]))
        numerator_sum = numerator_sum[i:]

    # NUMERATOR /DENOMINATOR
    r_hat = []
    for i in range(len(numerator)):
        r_hat.append([i, numerator[i] / denom])
        r_hat.append([i * -1, numerator[i] / denom])
    r_hat.pop(0)

    r_hat = pd.DataFrame([item[1] for item in r_hat], index=[item[0] for item in r_hat])

    r_hat.sort_index(ascending=True, inplace=True)

    # PLOT
    insignif = 1.96 / len(T) ** 0.5

    plt.stem(r_hat.index, r_hat[0])
    plt.fill_between(r_hat.index, insignif, insignif * -1, color='b', alpha=.2)
    plt.title(title)
    plt.xlabel(f"Lags")
    plt.ylabel(f"Magnitude")
    return r_hat

r_hat1 = auto_correlation(q1,4,"Autocorrelation Function of Question 1")
plt.show()
print(r_hat1)

r_hat2 = auto_correlation(wn, 20,"Autocorrelation Function of White Noise")
plt.show()

# 4 lags
# r_hat(0): 0,1,2,3,4 0,1,2,3,4 = (0,0)+(1,1)+(2,2)+(3,3)+(4,4)
# r_hat(1): 1,2,3,4 0,1,2,3 = (1,0)+(2,1)+(3,2)+(4,3)
# r_hat(2): 2,3,4 0,1,2 = (2,0)+(3,1)+(4,2)
# r_hat(3): 3,4 0,1 = (3,0)+(4,1)
# r_hat(4): 4 0 = (4,0)

# 2 lags
# r_hat(0): 0,1,2,3,4 0,1,2,3,4 = (0,0)+(1,1)+(2,2)+(3,3)+(4,4)
# r_hat(1): 1,2,3,4 0,1,2,3 = (1,0)+(2,1)+(3,2)+(4,3)
# r_hat(2): 2,3,4 0,1,2 = (2,0)+(3,1)+(4,2)

# Question 4:

from pandas_datareader import data
import yfinance as yf

yf.pdr_override()


ticker = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP','MSFT']

def stonks_plot(ticker_symbol, lags):
    df = data.get_data_yahoo([stock for stock in ticker_symbol], start="2000-01-01", end="2023-02-01")
    df = df['Close']
    df.dropna(inplace=True)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Time-series of Stocks", fontsize =24)
    for i in range(len(df.columns)):
        ax = plt.subplot(3, 2, i + 1)
        plt.plot(df.iloc[:, i])
        plt.title(f"{df.columns[i]}")
        plt.tight_layout()
        plt.grid()
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
    plt.show()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Autocorrelation Function of Stocks", fontsize =24)
    for i in range(len(df.columns)):
        ax = plt.subplot(3, 2, i + 1)
        auto_correlation(df.iloc[:, i], lags, f"{df.columns[i]}")
        ax.set(xlabel=None, ylabel=None)
        plt.tight_layout()
        plt.grid()
        plt.xlabel('lags')
        plt.ylabel('Magnitude')
    plt.show()
    return df

df = stonks_plot(ticker, 50)

# Question 5:

from toolbox import *

ADF_Cal(wn)
ADF_Cal(df.loc[:,'YELP'])
ADF_Cal(df.loc[:,'AAPL'])




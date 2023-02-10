import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Question 2:

def white_noise():
    start_date = '01/01/2000'
    end_date = '31/12/2000'
    date = pd.date_range(start_date, end_date, periods=1000)
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


df = white_noise()

# Question 3

q1 = [3, 9, 27, 81, 243]

def auto_correlation(T, tau):
    # Y-BAR
    y_bar = np.average(T)

    # DENOMINATOR
    x = []
    for t in range(len(T)):
        x.append(np.sum((T[t] - y_bar) ** 2))
        denom = sum(x)

    # NUMERATOR     *** PROBLEM IS HERE ****
    numerator_sum = []
    count = 0
    for i in range(tau + 1):
        for j in range(i, len(T)):
            numerator_sum.append((T[j] - y_bar) * (T[j - count] - y_bar))
        count += 1

    numerator = []
    for i in range(len(T), len(T) - tau, -1):
        numerator.append(sum(numerator_sum[:i]))
        numerator_sum = numerator_sum[i:]

    # NUM/DENOM ##### NOT HERE
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



auto_correlation(df, 20)

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

ticker = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']


def stonks_plot(ticker_symbol, lags):
    df = data.get_data_yahoo([stock for stock in ticker_symbol], start="2000-01-01", end="2023-02-01")
    df = df['Close']
    df.dropna(inplace=True)

    plt.figure(figsize=(16, 12))
    for i in range(len(df.columns)):
        ax = plt.subplot(3, 2, i + 1)
        plt.plot(df.iloc[:, i])
        plt.title(f"{df.columns[i]}")
    plt.show()

    plt.figure(figsize=(16, 12))
    for i in range(len(df.columns)):
        ax = plt.subplot(3, 2, i + 1)
        auto_correlation(df.iloc[:, i], lags)
        plt.title(f"{df.columns[i]} autocorrelation: {lags} lags")
    plt.show()


stonks_plot(ticker, 50)
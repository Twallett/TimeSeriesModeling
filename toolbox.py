from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def csvloadandplot(filepath, timestep):
    x = pd.read_csv(filepath, index_col=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    x.index = pd.date_range(start=x.index[0], end=x.index[-1], periods=len(x))
    for i in range(len(x.columns)):
        ax.plot(x.index, x.iloc[:, i], label=x.columns[i])
        ax.set_xticklabels(x.index[0:len(x):timestep].month_name())
    plt.xticks(x.index[0:len(x):timestep])
    xaxis = input("Input x-axis label:")
    plt.xlabel(xaxis)  # For LAB 1: Date
    yaxis = input("Input y-axis label:")
    plt.ylabel(yaxis)  # For LAB 1: USD($)
    plt.title(f"{yaxis} vs. {xaxis}")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    return x

def summary_stats(x):
    for i in range(len(x.columns)):
        print(
            f"The {x.columns[i]} mean is: {x.iloc[:, i].mean()} and the variance is: {x.iloc[:, i].var()} with standard deviation: {x.iloc[:, i].std()} median: {x.iloc[:, i].median()}")

def cal_rolling_mean_var(x):
    lst = []

    for i in range(len(x.columns)):
        for n in range(len(x)):
            if n < len(x):
                lst.append([i, n, x.iloc[:, i].head(n).mean(), x.iloc[:, i].head(n).var()])

    df = pd.DataFrame(lst)
    df = df.set_index(0)

    for i in range(len(x.columns)):
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        fig.supxlabel("Samples")
        ax[0].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 1])
        ax[0].set_title(f"Rolling mean - {x.columns[i]}")
        ax[0].set_ylabel("Magnitude")
        ax[1].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 2])
        ax[1].set_title(f"Rolling variance - {x.columns[i]}")
        ax[1].set_ylabel("Magnitude")
        plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def auto_correlation(T, tau, title):
    if type(T) != list():
        T = list(T)

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

    markerline, stemline, baseline = plt.stem(r_hat.index, r_hat[0], basefmt='grey')
    plt.fill_between(r_hat.index, insignif, insignif * -1, color='b', alpha=.2)
    plt.title(title)
    markerline.set_markerfacecolor('red')
    markerline.set_markeredgecolor('red')
    plt.xlabel(f"Lags")
    plt.ylabel(f"Magnitude")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    return r_hat

def cal_rolling_mean_var(x):
    lst = []

    for i in range(len(x.columns)):
        for n in range(len(x)):
            if n < len(x):
                lst.append([i, n, x.iloc[:, i].head(n).mean(), x.iloc[:, i].head(n).var()])

    df = pd.DataFrame(lst)
    df = df.set_index(0)

    for i in range(len(x.columns)):
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        fig.supxlabel("Samples")
        ax[0].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 1])
        ax[0].set_title(f"Rolling mean - {x.columns[i]}")
        ax[0].set_ylabel("Magnitude")
        ax[1].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 2])
        ax[1].set_title(f"Rolling variance - {x.columns[i]}")
        ax[1].set_ylabel("Magnitude")
        plt.show()



# DATS 6313: Lab 1 - Tyler Wallett
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Question 1:

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


df = csvloadandplot(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv', 31)


# Question 2:

def summary_stats(x):
    for i in range(len(x.columns)):
        print(
            f"The {x.columns[i]} mean is: {x.iloc[:, i].mean()} and the variance is: {x.iloc[:, i].var()} with standard deviation: {x.iloc[:, i].std()} median: {x.iloc[:, i].median()}")


summary_stats(df)


# Question 3:

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
        ax[0].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 1])
        ax[0].set_title(f"Rolling mean - {x.columns[i]}")
        ax[0].set_ylabel("Magnitude")
        ax[1].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 2])
        ax[1].set_title(f"Rolling variance - {x.columns[i]}")
        ax[1].set_ylabel("Magnitude")
        plt.show()


cal_rolling_mean_var(df)

# Question 4:

print(
    f"Given that each of the previous graphs, for rolling mean and variance based on Sales, AdBudget and GDP, seem to converge towards an average value, as the sample size gradually increases through time, each feature appears to be mean and variance stationary.")

# Question 5:

from statsmodels.tsa.stattools import adfuller


def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


ADF_Cal(df.Sales)
ADF_Cal(df.AdBudget)
ADF_Cal(df.GDP)

print(
    f"Based on the ADF results, at a 95% confidence level, Sales and GDP do not have a unit root, however AdBudget does have a unit root. Meaning, Sales and GDP are mean stationary, and AdBudget is not mean stationary. Contrasted to the previous results, of question 4, Adbudget graph appears to be deceitful in terms of stationarity.")

# Question 6:

from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)


kpss_test(df.Sales)
kpss_test(df.AdBudget)
kpss_test(df.GDP)

print(
    f"Based on the resulting 0.1 p-value for each kpss test, for Sales, AdBudget and GDP, we accept the null hypothesis that all features are mean and variance stationary. Contrasted to the previous results of question 5, kpss p-value seems to contradict that of AdBudget.")

# Question 7:

df2 = csvloadandplot(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv',
    12)

summary_stats(df2)

cal_rolling_mean_var(df2)

ADF_Cal(df2)

kpss(df2)

# Question 8:
import numpy as np


def ts_transformations(x, features_amount):
    if features_amount == 1:

        for i in range(len(x.columns)):
            diff1 = x.diff()
            plt.hist(diff1)
            plt.title(f"First order non-seasonal differencing: {x.columns[i]}")
            plt.xlabel("First order residuals")
            plt.ylabel("Frequency")
            plt.show()

        for i in range(len(x.columns)):
            diff2 = x.diff(periods=2)
            plt.hist(diff2)
            plt.title(f"Second order non-seasonal differencing: {x.columns[i]}")
            plt.xlabel("Second order residuals")
            plt.ylabel("Frequency")
            plt.show()

        for i in range(len(x.columns)):
            diff3 = x.diff(periods=3)
            plt.hist(diff3)
            plt.title(f"Third order non-seasonal differencing: {x.columns[i]}")
            plt.xlabel("Third order residuals")
            plt.ylabel("Frequency")
            plt.show()

        for i in range(len(x.columns)):
            difflog = np.log(x).diff()
            plt.hist(difflog)
            plt.title(f"First order Logarithmic differencing: {x.columns[i]}")
            plt.xlabel("First order Log residuals")
            plt.ylabel("Frequency")
            plt.show()

        logdf = np.log(x).diff()
        for i in range(len(logdf.columns)):
            kpss_test(logdf.iloc[1:, i])
            ADF_Cal(logdf.iloc[1:, i])

    elif features_amount > 1:
        fig, ax = plt.subplots(1, features_amount, figsize=(10, 5))
        fig.suptitle("First order non-seasonal differencing")
        fig.supxlabel("First order residuals")
        fig.supylabel("Frequency")
        for i in range(len(x.columns)):
            diff = x.iloc[:, i].diff()
            ax[i].hist(diff)
            ax[i].set_title(f"{x.columns[i]}")

        fig, ax = plt.subplots(1, features_amount, figsize=(10, 5))
        fig.suptitle("Second order non-seasonal differencing")
        fig.supxlabel("Second order residuals")
        fig.supylabel("Frequency")
        for i in range(len(x.columns)):
            diff = x.iloc[:, i].diff(periods=2)
            ax[i].hist(diff)
            ax[i].set_title(f"{x.columns[i]}")

        fig, ax = plt.subplots(1, features_amount, figsize=(10, 5))
        fig.suptitle("Third order non-seasonal differencing")
        fig.supxlabel("Third order residuals")
        fig.supylabel("Frequency")
        for i in range(len(x.columns)):
            diff = x.iloc[:, i].diff(periods=3)
            ax[i].hist(diff)
            ax[i].set_title(f"{x.columns[i]}")

        fig, ax = plt.subplots(1, features_amount, figsize=(10, 5))
        fig.suptitle("First order Logarithmic differencing")
        fig.supxlabel("First order Log residuals")
        fig.supylabel("Frequency")
        for i in range(len(x.columns)):
            difflog = np.log(x.iloc[:, i]).diff()
            ax[i].hist(difflog)
            ax[i].set_title(f"{x.columns[i]}")

        logdf = np.log(x).diff()
        for i in range(len(logdf.columns)):
            kpss_test(logdf.iloc[1:, i])
            ADF_Cal(logdf.iloc[1:, i])

ts_transformations(df2, 1)


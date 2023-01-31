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


x = csvloadandplot(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv', 31)


# Question 2:

def summary_stats(df):
    for i in range(len(df.columns)):
        print(
            f"The {df.columns[i]} mean is: {df.iloc[:, i].mean()} and the variance is: {df.iloc[:, i].var()} with standard deviation: {df.iloc[:, i].std()} median: {df.iloc[:, i].median()}")


summary_stats(x)


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


cal_rolling_mean_var(x)

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


ADF_Cal(x.Sales)
ADF_Cal(x.AdBudget)
ADF_Cal(x.GDP)

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


kpss_test(x.Sales)
kpss_test(x.AdBudget)
kpss_test(x.GDP)

print(
    f"Based on the resulting 0.1 p-value for each kpss test, for Sales, AdBudget and GDP, we accept the null hypothesis that all features are mean and variance stationary. Contrasted to the previous results of question 5, kpss p-value seems to contradict that of AdBudget.")

# Question 7:

new = csvloadandplot(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv',
    12)

summary_stats(new)

cal_rolling_mean_var(new)

ADF_Cal(new)

kpss(new)

#Question 8:



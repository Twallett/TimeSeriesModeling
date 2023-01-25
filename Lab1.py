# DATS 6313: Lab 1 - Tyler Wallett
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Question 1:

def csvloadandplot(filepath, timestep):
    x = pd.read_csv(filepath, parse_dates=[0], index_col = 0)
    fig, ax = plt.subplots(figsize = (10,8))
    for i in range(len(x.columns)):
        ax.plot(x.index, x.iloc[:,i], label = x.columns[i])
    plt.xticks(x.index[0:len(x.index):timestep])
    xaxis = input("Input x-axis label:")
    plt.xlabel(xaxis) #For LAB 1: Date
    yaxis = input("Input y-axis label:")
    plt.ylabel(yaxis) #For LAB 1: USD($)
    plt.title(f"{yaxis} vs. {xaxis}")
    plt.legend(loc = "upper left")
    plt.show()
    return x

tute = csvloadandplot('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv', 31)

# Question 2:

def summary_stats(df):
    for i in range(len(df.columns)):
        print(f"The {df.columns[i]} mean is: {df.iloc[:,i].mean()} and the variance is: {df.iloc[:,i].var()} with standard deviation: {df.iloc[:,i].std()} median: {df.iloc[:,i].median()}")

summary_stats(tute)

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
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 1])
        ax[1].plot(df[df.index == i][1], df[df.index == i].iloc[0:len(x), 2])

cal_rolling_mean_var(tute)

# Question 4:

#print(f"comments on results")

# Question 5:

from statsmodels.tsa.stattools import adfuller

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

ADF_Cal(x.Sales)
ADF_Cal(x.AdBudget)
ADF_Cal(x.GDP)

# Question 6:

from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)

kpss_test(x.Sales)
kpss_test(x.AdBudget)
kpss_test(x.GDP)

# Question 7:

#csvloadandplot('https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/AirPassengers.csv',1)


# Question 8:




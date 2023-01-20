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

x = csvloadandplot('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv', 31)

# Question 2:

def summary_stats(df):
    for i in range(len(df.columns)):
        print(f"The {df.columns[i]} mean is: {df.iloc[:,i].mean()} and the variance is: {df.iloc[:,i].var()} with standard deviation: {df.iloc[:,i].std()} median: {df.iloc[:,i].median()}")

summary_stats(x)

# Question 3:


# Question 4:


# Question 5:


# Question 6:


# Question 7:

#csvloadandplot('https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/AirPassengers.csv',1)

# Question 8:


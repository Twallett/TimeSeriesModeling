# DATS 6313: Lab 1 - Tyler Wallett

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Question 1:

def load(filepath):
    import pandas
    x = pd.read_csv(filepath, index_col = 0)
    return x

df = load('https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/tute1.csv')
print(df.columns)



# Question 2:



# Question 3:
# Question 4:
# Question 5:
# Question 6:
# Question 7:
# Question 8:


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = [[1, 112], [2, 118], [3, 132], [4, 129], [5, 121], [6, 135], [7, 148], [8, 136], [9, 119], [10, 104], [11, 118],
        [12, 115], [13, 126], [14, 141]]

data = pd.DataFrame([data[i][1] for i in range(len(data))], index=[data[i][0] for i in range(len(data))], columns=['y'])


# %%
# Question 2:

def average_method(df, h_step):
    # ONE STEP PREDICTION
    y_hat = []
    error = []
    for i in range(1, len(df[0:h_step]) + 1):
        y_hat.append(sum(df.iloc[:i, 0]) / i)
        error.append(df.iloc[i - 1, 0] - (sum(df.iloc[:i, 0]) / i))

    # H STEP PREDICTION
    y_hat_h = []
    error_h = []

    y_hat_h = y_hat[-1:] * len(df[h_step:])

    for i in range(h_step, len(df[h_step:]) + h_step):
        error_h.append(df.iloc[i, 0] - y_hat_h[i - h_step])

    y_hat = pd.Series(y_hat, name='y_hat')
    y_hat_h = pd.Series(y_hat_h, name='y_hat_h')
    error = pd.Series(error, name='error')
    error_h = pd.Series(error_h, name='error_h')

    y_hat = y_hat.append(y_hat_h)
    y_hat.index = np.arange(1, len(df) + 1)
    y_hat = pd.Series(y_hat, name='y_hat')

    error = error.append(error_h)
    error.index = np.arange(1, len(df) + 1)
    error = pd.Series(error, name='error')

    df = df.merge(y_hat, left_index=True, right_index=True)
    df = df.merge(error, left_index=True, right_index=True)
    df['error_sq'] = df.error ** 2

    plt.figure(figsize=(8, 5))
    plt.plot(df.iloc[0:h_step, 0], label='Training dataset')
    plt.plot(df.iloc[h_step:, 0], label='Testing dataset')
    plt.plot(df.iloc[h_step:, 1], label='Avg. method H-step prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('y-values')
    plt.title('Average method & forecast')
    plt.grid()
    plt.tight_layout()
    return df


newdata = average_method(data, 9)


# %%
# Question #3:

def mse_errors(df):
    mse_pred = df.iloc[2:9, 3].mean()
    mse_forecast = df.iloc[9:, 3].mean()
    return mse_pred, mse_forecast


mse_errors(newdata)


# %%
# Question #4:

def var_errors(df):
    var_pred = df.iloc[2:9, 3].var()
    var_forecast = df.iloc[9:, 3].var()
    return var_pred, var_forecast


var_errors(newdata)

# %%
# Question #5:
# def q_value(df):

from toolbox import *

auto_correlation(newdata.iloc[:9, 2], 5, '')


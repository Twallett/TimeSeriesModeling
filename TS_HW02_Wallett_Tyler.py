import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from toolbox import *

data = [[1, 112], [2, 118], [3, 132], [4, 129], [5, 121], [6, 135], [7, 148], [8, 136], [9, 119], [10, 104], [11, 118],
        [12, 115], [13, 126], [14, 141]]

data = pd.DataFrame([data[i][1] for i in range(len(data))], index=[data[i][0] for i in range(len(data))], columns=['y'])

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

    plt.plot(df.iloc[0:h_step, 0], label='Training dataset')
    plt.plot(df.iloc[h_step:, 0], label='Testing dataset')
    plt.plot(df.iloc[h_step:, 1], label='Avg. H-step prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('y-values')
    plt.title('Average method & forecast')
    plt.grid()
    plt.tight_layout()
    return df

newdata_avg = average_method(data, 9)
plt.show()
print('Average method:')
print(newdata_avg)

# Question 3:

def mse_errors(df):
    mse_pred = df.iloc[2:9, 3].mean()
    mse_forecast = df.iloc[9:, 3].mean()
    return mse_pred, mse_forecast

mse_pred_avg, mse_forecast_avg = mse_errors(newdata_avg)
print(f"MSE predicted Average method: {mse_pred_avg}")
print(f"MSE forecasted Average method: {mse_forecast_avg}")

# Question 4:

def var_errors(df):
    var_pred = df.iloc[2:9, 3].var()
    var_forecast = df.iloc[9:, 3].var()
    return var_pred, var_forecast

var_pred_avg,  var_forecast_avg = var_errors(newdata_avg)
print(f"VAR predicted Average method: {var_pred_avg}")
print(f"VAR forecasted Average method: {var_forecast_avg}")

# Question 5:

def q_value(df, lags):
    train = list(df)
    r_hat = auto_correlation(train, lags, '')
    r_hat = r_hat[lags + 1:lags * 2 + 1] ** 2
    q = ((np.cumsum(r_hat)) * 9).sum()[0]
    plt.close()
    return q


q_avg = q_value(newdata_avg.iloc[2:9, 2], 5)
print(f"Q-value Average method: {q_avg}")

# Question 6: Naive method 1-5

def naive_method(df, h_step):
    # ONE STEP PREDICTION
    y_hat = []
    for i in range(1, len(df[0:h_step]) + 1):
        if i == 1:
            y_hat.append(int(df.iloc[i - 1:i, 0]))
        if i > 1:
            y_hat.append(int(df.iloc[i - 2:i - 1, 0]))
        if i == h_step:
            y_hat.append(int(df.iloc[h_step - 1:h_step, 0]))

    # H STEP PREDICTION
    y_hat_h = []
    y_hat_h = y_hat[-1:] * len(df[h_step:])

    y_hat = pd.Series(y_hat, name='y_hat')
    y_hat_h = pd.Series(y_hat_h, name='y_hat_h')

    y_hat = y_hat.append(y_hat_h)
    y_hat.index = np.arange(1, len(df) + 2)
    y_hat = pd.Series(y_hat, name='y_hat')

    df = df.merge(y_hat, left_index=True, right_index=True)
    df['error'] = df.y - df.y_hat
    df['error_sq'] = df.error ** 2

    plt.plot(df.iloc[0:h_step, 0], label='Training dataset')
    plt.plot(df.iloc[h_step:, 0], label='Testing dataset')
    plt.plot(df.iloc[h_step:, 1], label='Naive H-step prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('y-values')
    plt.title('Naive method & forecast')
    plt.grid()
    plt.tight_layout()
    return df

newdata_naive = naive_method(data, 9)
plt.show()

mse_pred_naive, mse_forecast_naive = mse_errors(newdata_naive)
print(f"Naive method:")
print(newdata_naive)
print(f"MSE Predicted Naive method: {mse_pred_naive}")
print(f"MSE Forecasted Naive method: {mse_forecast_naive}")

var_pred_naive,  var_forecast_naive = var_errors(newdata_naive)
print(f"VAR Predicted Naive method: {var_pred_naive}")
print(f"VAR Forecasted Naive method: {var_forecast_naive}")

q_naive = q_value(newdata_naive.iloc[2:9, 2], 5)
print(f"Q-value Naive method: {q_naive}")

# Question 7: Drift method 1-5

def drift_method(df, h_step):
    # ONE STEP PREDICTION
    y_hat = []
    for i in range(1, len(df) + 1):
        if i == 1:
            y_hat.append(int(df.iloc[i - 1:i, 0]))
        if i == 2:
            y_hat.append(int(df.iloc[i - 2:i - 1, 0]))
        if 2 < i <= len(df[0:h_step]):
            y_hat.append(df.iloc[i - 1, 0] + i * ((df.iloc[i - 1, 0] - df.iloc[0, 0]) / (i - 1)))
        if i > len(df[0:h_step]):
            y_hat.append(df.iloc[h_step - 1, 0] + (i - h_step) * (
                        (df.iloc[h_step - 1, 0] - df.iloc[0, 0]) / (len(df.iloc[0:h_step, 0]) - 1)))

    # H STEP PREDICTION

    y_hat = pd.Series(y_hat, name='y_hat')

    y_hat.index = np.arange(1, len(df) + 1)
    y_hat = pd.Series(y_hat, name='y_hat')

    df = df.merge(y_hat, left_index=True, right_index=True)
    df['error'] = df.y - df.y_hat
    df['error_sq'] = df.error ** 2

    plt.plot(df.iloc[0:h_step, 0], label='Training dataset')
    plt.plot(df.iloc[h_step:, 0], label='Testing dataset')
    plt.plot(df.iloc[h_step:, 1], label='Drift H-step prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('y-values')
    plt.title('Drift method & forecast')
    plt.grid()
    plt.tight_layout()
    return df


newdata_drift = drift_method(data, 9)
plt.show()

mse_pred_drift, mse_forecast_drift = mse_errors(newdata_drift)
print('Drift method:')
print(newdata_drift)
print(f"MSE predicted Drift method: {mse_pred_drift}")
print(f"MSE forecasted Drift method: {mse_forecast_drift}")

var_pred_drift,  var_forecast_drift = var_errors(newdata_drift)
print(f"VAR predicted Drift method: {var_pred_drift}")
print(f"VAR forecasted Drift method: {var_forecast_drift}")

q_drift = q_value(newdata_drift.iloc[2:9, 2], 5)
print(f"Q-value Drift method: {q_drift}")

# Question 8: SES aplha 0.5

def ses_method(df, h_step, alpha):
    # ONE STEP PREDICTION
    y_hat = []
    for i in range(1, len(df) + 1):
        if i == 1:
            y_hat.append(df.iloc[0, 0])
        if 1 < i <= h_step + 1:
            y_hat.append((alpha * df.iloc[i - 2, 0]) + ((1 - alpha) * y_hat[i-2]))
        if i > h_step + 1:
            y_hat.append(y_hat[h_step])

    # H STEP PREDICTION

    y_hat = pd.Series(y_hat, name='y_hat')

    y_hat.index = np.arange(1, len(df) + 1)
    y_hat = pd.Series(y_hat, name='y_hat')

    df = df.merge(y_hat, left_index=True, right_index=True)
    df['error'] = df.y - df.y_hat
    df['error_sq'] = df.error ** 2

    plt.plot(df.iloc[0:h_step, 0], label='Training dataset')
    plt.plot(df.iloc[h_step:, 0], label='Testing dataset')
    plt.plot(df.iloc[h_step:, 1], label='SES H-step prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('y-values')
    plt.title(f'SES method & forecast (alpha: {alpha})')
    plt.grid()
    plt.tight_layout()
    return df

newdata_ses = ses_method(data, 9, 0.5)
plt.show()

mse_pred_ses, mse_forecast_ses = mse_errors(newdata_ses)
print('SES method (alpha 0.5):')
print(newdata_ses)
print(f"MSE predicted SES method (alpha 0.5): {mse_pred_ses}")
print(f"MSE forecasted SES method (alpha 0.5): {mse_forecast_ses}")

var_pred_ses, var_forecast_ses = var_errors(newdata_ses)
print(f"VAR predicted SES method (alpha 0.5): {var_pred_ses}")
print(f"VAR forecasted SES method (alpha 0.5): {var_forecast_ses}")

q_ses = q_value(newdata_ses.iloc[2:9, 2], 5)
print(f"Q-value SES method (alpha 0.5): {q_ses}")

# Question 9: SES alpha 0, 0.25, 0.75, 0.99

def ses_subplots(df, h_step, *alphas):
    a_list = [*alphas]
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("SES subplot", fontsize=24)
    for i in range(len(a_list)):
        ax = plt.subplot(2, 2, i + 1)
        ses_method(df, h_step, a_list[i])
        ax.set(xlabel=None, ylabel=None)
        plt.title(f"SES alpha {a_list[i]}")
        plt.tight_layout()
        plt.xlabel('Date')
        plt.ylabel('y-values')
    plt.show()
    return a_list

ses_subplots(data, 9, 0, 0.25, 0.5, 0.99)

# Question 10:

q_values = [q_avg, q_naive, q_drift, q_ses]

mse_preds = [mse_pred_avg, mse_pred_naive, mse_pred_drift, mse_pred_ses]

mse_forecasts = [mse_forecast_avg, mse_forecast_naive, mse_forecast_drift, mse_forecast_ses]

mresiduals = [newdata_avg.iloc[2:9,2].mean(), newdata_naive.iloc[2:9,2].mean(), newdata_drift.iloc[2:9,2].mean(), newdata_ses.iloc[2:9,2].mean()]

var_preds = [var_pred_avg, var_pred_naive, var_pred_drift, var_pred_ses]

df_q10 = pd.DataFrame([q_values, mse_preds, mse_forecasts, mresiduals, var_preds], columns=['Average_method', 'Naive_method', 'Drift_method', 'SES_method_0.5'], index= ['q_values', 'mse_predictions', 'mse_forecasts', 'mean_res', 'var_pred_res'])

print(df_q10)

# Question 11:

auto_correlation(newdata_avg.iloc[2:9,2], 6, 'Average method autocorrelation of residuals')
plt.show()

auto_correlation(newdata_naive.iloc[2:9,2], 6, 'Naive method autocorrelation of residuals')
plt.show()

auto_correlation(newdata_drift.iloc[2:9,2], 6, 'Drift method autocorrelation of residuals')
plt.show()

auto_correlation(newdata_ses.iloc[2:9,2], 6, 'SES method autocorrelation of residuals (alpha 0.5)')
plt.show()

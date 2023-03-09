# %%
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import *

# Question 1:

data = pd.read_csv(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv',
    index_col=0)

data.index = pd.date_range(start='1981-01-01', periods=len(data), freq='D')


def moving_average(df):
    m = int(input("Input m order:"))

    if m == 1 or m == 2:
        raise ValueError("m=1,2 will not be accepted.")

    if m % 2 != 0:
        k = int((m - 1) / 2)

        error_l = []

        for i in range(len(df) + 1):
            if i < k:
                error_l.append(0)
            if k <= i < len(df) - k:
                error_l.append((1 / m) * df[i - k:i + k + 1].sum()[0])
            if len(df) - k < i < len(df) + 1:
                error_l.append(0)

        df[f"{m}m"] = error_l

    if m % 2 == 0:

        folding_order = int(input("Input folding order:"))
        k = int((m - 2) / 2)

        ma_even = []
        for i in range(len(df) + 1):
            if i < k:
                ma_even.append(0)
            if k <= i < len(df) - k - 1:
                ma_even.append((1 / m) * df[i - k:i + k + 2].sum()[0])
            if len(df) - k - 1 <= i < len(df):
                ma_even.append(0)

        ma_even_folding = []
        for i in range(len(ma_even)):
            if i < k + 1:
                ma_even_folding.append(0)
            if k + 1 <= i < len(ma_even) - k - 1:
                ma_even_folding.append((1 / folding_order) * sum(ma_even[i - 1: i + 1]))
            if len(ma_even) - k - 1 <= i < len(ma_even):
                ma_even_folding.append(0)

        df[f"{m}m"] = ma_even_folding
    return df


# %%

# Question 2:

m3 = moving_average(data)

m5 = moving_average(data)

m7 = moving_average(data)

m9 = moving_average(data)

# %%

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(m3.index[:50], m3.Temp[:50])
ax[0, 0].plot(m3.index[1:51], m3["3m"][1:51])
ax[0, 0].set_title("3m moving average")
ax[0, 0].grid()
ax[0, 0].set_xlabel("Dates")
ax[0, 0].set_ylabel("Temperature")
ax[0, 0].legend(["Original", "3M"])

ax[0, 1].plot(m3.index[:50], m3.Temp[:50])
ax[0, 1].plot(m3.index[2:52], m3["5m"][2:52])
ax[0, 1].set_title("5m moving average")
ax[0, 1].grid()
ax[0, 1].set_xlabel("Dates")
ax[0, 1].set_ylabel("Temperature")
ax[0, 1].legend(["Original", "5M"])

ax[1, 0].plot(m3.index[:50], m3.Temp[:50])
ax[1, 0].plot(m3.index[3:53], m3["7m"][3:53])
ax[1, 0].set_title("7m moving average")
ax[1, 0].grid()
ax[1, 0].set_xlabel("Dates")
ax[1, 0].set_ylabel("Temperature")
ax[1, 0].legend(["Original", "7M"])

ax[1, 1].plot(m3.index[:50], m3.Temp[:50])
ax[1, 1].plot(m3.index[4:54], m3["9m"][4:54])
ax[1, 1].set_title("9m moving average")
ax[1, 1].grid()
ax[1, 1].set_xlabel("Dates")
ax[1, 1].set_ylabel("Temperature")
ax[1, 1].legend(["Original", "9M"])

for i in range(2):
    for j in range(2):
        ax[i, j].set_xticks(m3.index[:50:10])
        ax[i, j].set_xticklabels(m3.index[:50:10].strftime('%b %d'))

plt.tight_layout()
plt.show()

# %%

# Question 3:

m4 = moving_average(data)

m6 = moving_average(data)

m8 = moving_average(data)

m10 = moving_average(data)

# %%

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(m4.index[:50], m4.Temp[:50])
ax[0, 0].plot(m4.index[2:52], m4["4m"][2:52])
ax[0, 0].set_title("2X4m moving average")
ax[0, 0].grid()
ax[0, 0].set_xlabel("Dates")
ax[0, 0].set_ylabel("Temperature")
ax[0, 0].legend(["Original", "2X4M"])

ax[0, 1].plot(m4.index[:50], m4.Temp[:50])
ax[0, 1].plot(m4.index[3:53], m4["6m"][3:53])
ax[0, 1].set_title("2X6m moving average")
ax[0, 1].grid()
ax[0, 1].set_xlabel("Dates")
ax[0, 1].set_ylabel("Temperature")
ax[0, 1].legend(["Original", "2X6M"])

ax[1, 0].plot(m4.index[:50], m4.Temp[:50])
ax[1, 0].plot(m4.index[4:54], m4["8m"][4:54])
ax[1, 0].set_title("2X8m moving average")
ax[1, 0].grid()
ax[1, 0].set_xlabel("Dates")
ax[1, 0].set_ylabel("Temperature")
ax[1, 0].legend(["Original", "2X8M"])

ax[1, 1].plot(m4.index[:50], m4.Temp[:50])
ax[1, 1].plot(m4.index[5:55], m4["10m"][5:55])
ax[1, 1].set_title("2X10m moving average")
ax[1, 1].grid()
ax[1, 1].set_xlabel("Dates")
ax[1, 1].set_ylabel("Temperature")
ax[1, 1].legend(["Original", "2X10M"])

for i in range(2):
    for j in range(2):
        ax[i, j].set_xticks(m4.index[:50:10])
        ax[i, j].set_xticklabels(m4.index[:50:10].strftime('%b %d'))

plt.tight_layout()
plt.show()

# %%

# Question 4:

ADF_Cal(m4.Temp[:51])

ADF_Cal(m4["3m"][1:51])

# %%

# Question 5:

from statsmodels.tsa.seasonal import STL

stl = STL(data['Temp'])
res = stl.fit()
fig = res.plot()
plt.legend()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(8, 6))
plt.title("Seasonality, Trend and Residual components")
plt.plot(T)
plt.plot(S)
plt.plot(R)
plt.xlabel("Dates")
plt.ylabel("Temperature")
plt.legend(["Trend", "Seasonality", "Residuals"])
plt.show()

# %%

# Question 6:

plt.figure(figsize=(8, 6))
plt.plot(data['Temp'])
plt.plot(S)
plt.grid()
plt.title("Temperature and seasonality component")
plt.xlabel("Dates")
plt.ylabel("Temperature")
plt.legend(["Original Data", "Seasonality Component"])
plt.show()


# %%

# Question 7:

# Strength of trend

def t_strength(R_STL, T_STL):
    t_strength = 1 - ((R.var(ddof=0)) / (T + R).var(ddof=0))
    return print(f"The strength of trend for this data set is {t_strength}")


t_strength(R, T)


# %%

# Question 8:

# Strength of seasonality

def s_strength(R_STL, S_STL):
    s_strength = 1 - ((R.var(ddof=0)) / (S + R).var(ddof=0))
    print(f"The strength of seasonality for this data set is {s_strength}")


s_strength(R, S)

# %%

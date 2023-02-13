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

from statsmodels.tsa.stattools import adfuller

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
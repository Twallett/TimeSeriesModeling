import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Question 2:

def white_noise():
    start_date = '01/01/2000'
    end_date = '31/12/2000'
    date = pd.date_range(start_date, end_date, periods=1000)
    x = np.random.normal(0, 1, size=1000).round(2)
    df = pd.DataFrame(x, index=date)
    plt.hist(df)
    plt.xlabel('White noise standard deviations')
    plt.ylabel('Frequency')
    plt.title('Histogram of White Noise')
    plt.show()
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('White Noise')
    plt.title('White Noise Time-series')
    plt.show()
    print(f"Sample mean: {x.mean()}")
    print(f"Sample std: {x.std()}")

white_noise()

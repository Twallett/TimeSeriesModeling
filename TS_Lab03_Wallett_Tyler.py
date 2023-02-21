# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

# Question 1:

train_size = 0.8

data = pd.read_csv(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/autos.clean.csv')

data = data.loc[:, ['price', 'normalized-losses', 'wheel-base', 'length', 'width',
                    'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower',
                    'peak-rpm', 'city-mpg', 'highway-mpg']]


def train_test_split(df, train_size):
    train_length = round(train_size * len(df))
    test_length = round(len(df) - train_size * len(df))

    train = df.iloc[:train_length, :]
    test = df.iloc[train_length:, :]

    print(f'Length of dataset is {len(data)}', '\n')
    print(f'Train length is {train_length}', '\n')
    print(f'Test length is {test_length}', '\n')
    return train, test


train, test = train_test_split(data, 0.8)

# %%

# Question 2:

plt.figure(figsize=(9, 7))

mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

datacorr = data.corr()

sns.heatmap(datacorr,
            annot=True,
            mask=mask)

plt.tight_layout()

# %%

# Question 3:

# Create a matrix X mxn where m is the amount of obs and n features

X = np.matrix(data.loc[:, data.columns != 'price'])

H = X.T @ X

s, d, v = LA.svd(H)

condition_number = H[0, 0] / H[0, -1]

print("SingularValues = ", d, '\n')

print("Condition number =", condition_number)
# %%

# Question 4:
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

# Question 1:

train_size = 0.8

data = pd.read_csv(
    'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/autos.clean.csv')

data = data.loc[:, ['price', 'normalized-losses', 'wheel-base', 'length', 'width',
                    'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower',
                    'peak-rpm', 'city-mpg', 'highway-mpg']]


def train_test_split(df, train_size):
    train_length = round(train_size * len(df))
    test_length = round(len(df) - train_size * len(df))

    train = df.iloc[:train_length, :]
    test = df.iloc[train_length:, :]

    print(f'Length of dataset is {len(data)}', '\n')
    print(f'Train length is {train_length}', '\n')
    print(f'Test length is {test_length}', '\n')
    return train, test


train, test = train_test_split(data, 0.8)

# %%

# Question 2:

plt.figure(figsize=(9, 7))

mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

datacorr = data.corr()

sns.heatmap(datacorr,
            annot=True,
            mask=mask)

plt.tight_layout()

# %%

# Question 3:

# Create a matrix X mxn where m is the amount of obs and n features

X = np.matrix(data.loc[:, data.columns != 'price'])

H = X.T @ X

s, d, v = LA.svd(H)

condition_number = H[0, 0] / H[0, -1]

print("SingularValues = ", d, '\n')

print("Condition number =", condition_number)
# %%

# Question 4:

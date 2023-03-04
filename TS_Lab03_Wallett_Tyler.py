import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from statsmodels.regression import linear_model as lm
import statsmodels.api as sm
from toolbox import *
from sklearn.preprocessing import StandardScaler

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

# Question 2:

plt.figure(figsize=(9, 7))

mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

datacorr = data.corr()

sns.heatmap(datacorr,
            annot=True,
            mask=mask)

plt.tight_layout()
plt.title("Correlation Plot of Auto data")
plt.show()

# Question 3:

# Create a matrix X mxn where m is the amount of obs and n features

X = data.loc[:, data.columns != 'price']

H = X.T @ X

s, d, v = LA.svd(H)

condition_number = LA.cond(X)

print("SingularValues = ", d, '\n')

print("Condition number =", condition_number)

X = X.drop(columns=["width"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["curb-weight"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["bore"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["stroke"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["peak-rpm"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["city-mpg"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["length"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

X = X.drop(columns=["height"])

new_condition_number = LA.cond(X)

print("New Condition number =", new_condition_number)

# Question 4:

train.drop(columns=["width", "curb-weight", "bore", "stroke", "peak-rpm", "city-mpg", "length", "height"], inplace=True)

scaler = StandardScaler()

data_stan = pd.DataFrame(scaler.fit_transform(train),
                         columns=["price", "normalized-losses", "wheel-base", "engine-size", "compression-ratio",
                                  "horsepower", "highway-mpg"])

# Question 5:

# X, Y standardized matrices

X_stan = np.matrix(data_stan.iloc[:, data_stan.columns != 'price'])
ones = np.ones((161, 1))

X_stan = np.hstack((ones, X_stan))
Y_stan = np.matrix(data_stan.iloc[:, data_stan.columns == 'price'])

beta_hat_stan = LA.inv(X_stan.T @ X_stan) @ X_stan.T @ Y_stan

print(beta_hat_stan)

# Question 6: Compare results of 4 and 5

model = lm.OLS(Y_stan, X_stan)
results = model.fit()

ols_coefficients = results.params
print(ols_coefficients)

print(results.summary())

# Question 7:

# AIC, BIC and ADJ R^2 Backward stepwise regression

X_stan = pd.DataFrame(data_stan.loc[:, data_stan.columns != 'price'])
Y_stan = pd.DataFrame(data_stan.loc[:, data_stan.columns == 'price'])
X_stan = sm.add_constant(X_stan)
def backward_stepwise_regression(X, Y):
    bic_ = []
    aic_ = []
    r_2_ = []

    for i in range(len(X.columns), 0, -1):
        model = lm.OLS(Y, X.iloc[:, :i])
        results = model.fit()
        bic = results.bic
        aic = results.aic
        r_2 = results.rsquared_adj
        bic_.append(bic)
        aic_.append(aic)
        r_2_.append(r_2)
        print(f"BIC: {round(bic, 2)}, AIC: {round(aic, 2)}, r_2: {round(r_2, 4)}.")
        count = 0
        if (bic < bic_[count]) and (aic < aic_[count]) and (r_2 > r_2_[count]):
            count += 1
            print(f"Remove the last {len(X.columns) - i} features: {list(X.columns[-(len(X.columns) - i):])}")
            break
        else:
            continue

backward_stepwise_regression(X_stan, Y_stan)

# (OPTIONAL) Question 8:

# AIC, BIC and ADJ R^2 Forward Stepwise Regression
def forward_stepwise_regression(X, Y):
    bic_ = []
    aic_ = []
    r_2_ = []
    for i in range(1, len(X.columns)):
        model = lm.OLS(Y, X.iloc[:, :i])
        results = model.fit()
        bic = results.bic
        aic = results.aic
        r_2 = results.rsquared_adj
        bic_.append(bic)
        aic_.append(aic)
        r_2_.append(r_2)
        print(f"BIC: {round(bic, 2)}, AIC: {round(aic, 2)}, r_2: {round(r_2, 4)}.")
        count = 0
        if i > 2:
            if (bic < bic_[count]) and (aic < aic_[count]) and (r_2 > r_2_[count]):
                count += 1
            print(f"Keep the first {i-1} features: {list(X.columns[:i])}")
            break
        else:
            continue

forward_stepwise_regression(X_stan, Y_stan)

# Question 10:

finalmodel = lm.OLS(Y_stan, X_stan.iloc[:, :-1])
finalresults = finalmodel.fit()
print(finalresults.summary())

# Question 11:

data_test_stan = pd.DataFrame(scaler.fit_transform(test),
                              columns=["price", "normalized-losses", "wheel-base", "length", "width", "height",
                                       "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
                                       "horsepower", "peak-rpm", "city-mpg", "highway-mpg"],
                              index=[i for i in range(161, 201)])

data_test_stan.drop(
    columns=["width", "curb-weight", "bore", "stroke", "peak-rpm", "city-mpg", "length", "height", "highway-mpg"],
    inplace=True)

beta_hat_stan = np.resize(beta_hat_stan, (6, 1))

predictors = data_test_stan.loc[:, data_test_stan.columns != 'price']

predictors = sm.add_constant(predictors)

predictions = predictors @ beta_hat_stan

data_test_stan['predictions'] = predictions

plt.plot(data_stan.loc[:, 'price'])
plt.plot(data_test_stan.loc[:, 'price'])
plt.plot(data_test_stan.loc[:, 'predictions'])
plt.xlabel('Number of observations')
plt.ylabel('Standardized values')
plt.title('Standardized values vs number of observations')
plt.tight_layout()
plt.show()

# Question 12:

# ACF of errors 20 lags

data_test_stan['error'] = data_test_stan.loc[:, 'price'] - data_test_stan.loc[:, 'predictions']

auto_correlation(data_test_stan.loc[:, 'error'], 20, 'Autocorrelation of errors: 20 lags')
plt.show()

# Question 13:

# f test

A = np.identity(len(finalresults.params))
A = A[1: , :]
print(finalresults.f_test(A))

# t test
B = np.identity(len(finalresults.params))
B = B[1: , :]
print(finalresults.t_test(B))

# %%

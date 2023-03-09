import numpy as np

np.random.seed(6313)
mean = 1
std = 1
n = 1000
e = np.random.normal(mean, std, n)

# y(t) + 0.5y(y-1) + 0.25y(t-2) = e(t)

# Method 1: Simulate AR process using for loop

y = np.zeros(len(e))

for i in range(len(e)):
    if i == 0:
        y[0] = e[0]
    elif i == 1:
        y[i] = -0.5*y[i-1] + e[i]
    else:
        y[i] = -0.5 * y[i - 1] - 0.25 * y[i - 2] + e[i]

print(f"For loop method {y[:3]}")

# Method 2: dlsim method

from scipy import signal

num = [1,0.25,0.5]
den = [1,0,0]

system = (num, den, 1)
t, y_dlsim = signal.dlsim(system, e)

print(f'y(dlsim) {y_dlsim[:3]}')
print(f'the experimental mean of y is {np.mean(y_dlsim)}')





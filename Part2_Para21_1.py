import numpy as np
from math import *
from scipy.stats import beta, bernoulli
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
a, b = 1, 1
x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
plt.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')

r = np.random.random(1000)
for idx in range(0, len(r)):
    current_toss = r[idx]
    print("Try %d, value = %f" % (idx, current_toss))
    if current_toss > 0.3:
        a += 1
    if current_toss <= 0.3:
        b += 1
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    plt.cla()
    plt.plot(x, beta.pdf(x, a, b), 'r-', lw=3, alpha=0.6, label='beta pdf')
    plt.pause(0.05)



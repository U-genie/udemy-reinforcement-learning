import numpy as np
from math import *
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m
        self.mean = upper_limit
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N+=1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N * x


def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1, 4.0), Bandit(m2, 4.0), Bandit(m3, 4.0)]

    data = np.empty(N)
    for i in range(1, N+1):
        # implementing UCB1
        j = np.argmax([b.mean + sqrt(2.0*log(i)/(b.N+0.000001)) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i-1] = x
    cumulative_average = np.cumsum(data) / (np.arange(N)+1)
    '''
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale("log")
    plt.show()
    '''
    for b in bandits:
        print(b.mean)
    return cumulative_average

if __name__ == "__main__":
    m_1 = 1.0
    m_2 = 2.0
    m_3 = 3.0
    N = 100000
    c_1 = run_experiment(m_1, m_2, m_3, N)
    c_05 = run_experiment(m_1, m_2, m_3, N)
    c_01 = run_experiment(m_1, m_2, m_3, N)
    #log scale plot
    plt.plot(np.ones(N) * m_1)
    plt.plot(np.ones(N) * m_2)
    plt.plot(np.ones(N) * m_3)
    plt.plot(c_1, label="try 1")
    plt.plot(c_05, label="try 2")
    plt.plot(c_01, label="try 3")
    plt.legend()
    plt.title("Log plot")
    plt.xscale("log")
    plt.show()

    #linear plot
    plt.plot(c_1, label="try 1")
    plt.plot(c_05, label="try 2")
    plt.plot(c_01, label="try 3")
    plt.legend()
    plt.title("linear plot")
    plt.xscale("linear")
    plt.show()

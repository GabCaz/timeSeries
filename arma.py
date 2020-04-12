import statsmodels
import numpy as np
import pixiedust
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
class ARMA:
    '''ARMA class for processes of form:
       x(t) = c + ar[1] * x(t - 1) + ... + ar[p] * c(t - p)
                + ma[1] * eps(t - 1) + ... + ma[q] * eps (t - q) + eps(t)'''
    def __init__(self, ar = [], ma = [], constant = 0):
        ''' Pass in AR weights, MA weights, and constant '''
        self.ar = ar
        self.ma = ma
        self.c = constant

    def simulate(self, noise = None, length = 100):
        '''Simulate ARMA. Input: noise of the size of length'''
        if noise is None: # if no noise specified, make standard Gaussian noise
            noise = np.random.randn(length)
        p, q = len(self.ar), len(self.ma) # Orders
        maxOrder = max(p, q)
        # Add zeroes on the left of the noise to loop from beginning
        noise = np.concatenate((np.zeros(maxOrder), noise))
        # numpy arrays are reversed for easier indexing:
        ar, ma = np.array(self.ar[::-1]), np.array(self.ma[::-1])
        sim = np.zeros(length + maxOrder) # add zeros to not go out of bound when looping
        for i in range(maxOrder, maxOrder + length):
            sim[i] = np.sum(sim[i - p] * ar) + np.sum(noise[i - q] * ma) + noise[i] + self.c
        return sim[maxOrder:] # remove zero terms added and return

    def plotPath(self, noise = None, length = 100, ax = None, title = None):
        if ax is None:
            fig, ax = plt.subplots()
        sim = self.simulate(noise, length)
        ax.plot(range(length), sim)
        if title is None:
            title = 'T = ' + str(length) + ' for ARMA (' + str(len(self.ar)) + ',' + str(len(self.ma)) + ')'
        plt.title(title)
        plt.xlabel('t (time)')
        plt.ylabel('x (value)')
        plt.grid()
        return sim

    def plotIrf(self, length = 30, ax = None):
        ''' Plot the impulse response function of the process '''
        noise = np.concatenate((np.array([1]), np.zeros(length - 1)))
        title = 'IRF for ARMA (' + str(len(self.ar)) + ',' + str(len(self.ma)) + ')'
        maxiVal = max(self.plotPath(noise, length, title = title))
        plt.ylim((-0.2, max(1., maxiVal) + 0.1))

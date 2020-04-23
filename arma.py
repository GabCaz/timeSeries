import statsmodels
import pdb
import numpy as np
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

    def simulate(self, noise = None, length = 100, nsim = 1):
        '''Simulate ARMA.
        Input: noise of the size of length (by default, NWN)'''
        ar, ma = np.array(self.ar[::-1]), np.array(self.ma[::-1])
        p, q = len(ar), len(ma) # Orders
        max_order = max(p,q)

        if noise is None: # if no noise specified, make standard Gaussian noise
            noise = np.random.normal(size=(nsim, length))
        # Add zeroes for t < 0 to loop from beginning
        noise = np.concatenate((np.zeros([nsim, max_order]), noise), axis=1)
        # numpy arrays are reversed for easier indexing
        sim = np.zeros((nsim, length + max_order)) # add zeros to stay within bound when looping
        for i in range(max_order, max_order + length):
            sim[:, i] = np.sum(sim[:, i - p:i] * ar, axis=1) + np.sum(noise[:, i - q:i] * ma, axis=1) + noise[:, i] + self.c
        return sim[:, max_order:].T.squeeze() # remove zero terms added and return

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
        # make a no-constant ARMA to model IRF
        ir = ARMA(self.ar, self.ma)
        noise = np.concatenate((np.ones(shape = (1, 1)), np.zeros(shape = (1, length - 1))), axis = 1)
        title = 'IRF for ARMA (' + str(len(self.ar)) + ',' + str(len(self.ma)) + ')'
        maxiVal = max(ir.plotPath(noise, length, title = title))
        plt.ylim((-0.2, max(1., maxiVal) + 0.1))

if __name__ == '__main__':
    # pdb.set_trace()
    # q3 = ARMA(ar = [0.95, 0.9, 0.8])
    # print(q3.simulate(length = 400)[350])
    0

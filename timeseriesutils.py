# Utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
def plotDistrib(k, dictOfDistribs, add_normal=True):
    '''
    Purpose: given a number k and a dictionary of array of numbers, will plot the
    distribution of the array of numbers at key k
    If addNormalFit, will plot the fitted normal pdf
    '''
    bin_edges = np.linspace(min(dictOfDistribs[k]), max(dictOfDistribs[k]), num=50)
    plt.hist(dictOfDistribs[k], bins=bin_edges, density=True)
    plt.title('Distribution for sample size: ' + str(k))
    plt.xlabel('Values: x')
    plt.ylabel('Counts: f(x)')
    if add_normal:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, loc=np.mean(dictOfDistribs[k]), scale=np.std(dictOfDistribs[k]))
        plt.plot(x, p, 'k', linewidth=2, color="r")


def plotAllDistribs(dictOfDistribs, add_normal = True):
    '''
    Purpose: given a dictionary of array of numbers, will plot the distribution
    of each array
    '''
    plt.figure(figsize = [10, 6 * len(dictOfDistribs)]) # make large figure
    for k, i in zip(sorted(list(dictOfDistribs.keys())), range(1, len(dictOfDistribs) + 1)):
        plt.subplot(len(dictOfDistribs), 1, i)
        plotDistrib(k, dictOfDistribs, add_normal)

def getDimensions(X):
    '''retrieve number of observations and number of features'''
    if (X.ndim == 1):
        k = 1
        n = X.shape[0]
    else:
        n,k = X.shape
    return n, k

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from olsregr import OLSRegression
from arma import ARMA
class TimeSeries:
    '''
    A class to analyze time series
    Dependencies: OLS Regression
    Attributes:
    *** data: a numpy array representing a time series
    *** name (optional): name of the time series
    '''
    def __init__(self, data, name=None, time_ticks=None):
        '''data: a numpy array representing a time series'''
        self.data = data
        self.name = name
        self.time_ticks = time_ticks

    def plot_diff_data(self, diff=0):
        ''' plots the data differentiated diff (ie after applying
            delta ^ diff operator) '''
        if diff != 0:
            to_plot = self.data[:-diff].values - self.data[diff:].values
        else:
            to_plot = self.data
        if self.time_ticks is not None:
            plt.plot(self.time_ticks[diff:], to_plot)
        else:
            plt.plot(to_plot)
        if self.name is not None:
            plt.title(str(self.name) + ' differenced ' + str(diff) + ' times ')
        else:
            plt.title('time series differenced ' + str(diff) + ' times ')


    def getAcfLag(self, lag):
        ''' returns the lag-th ACF of the data.
            NB: This assumes stationarity (ACF depends only on the lag)'''
        if lag == 0:
            return 1.0
        return OLSRegression(self.data[:-lag].reshape((-1, 1)), self.data[lag:].reshape((-1, 1)), addConstant=True).beta_hat[-1]

    def getAcfUpToLag(self, lag):
        ''' returns all ACF values up to the given lag '''
        acf = []
        for i in range(lag):
            acf.append(self.getAcfLag(lag = i))
        return acf

    def plotAuto(self, values, lag, ax, title, xlabel = 'Time'):
        '''plot function used for acf and pacf'''
        if ax is None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (0.4 * lag, 5))
        ax.plot(range(lag), values, marker = 'o', linestyle = '--')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylim(-1.1, 1.1)
        # add vertical lines for readability
        for xc, yc in zip(range(lag), values):
            plt.plot([xc, xc], [0, yc], 'k-', lw = 1)

    def plotAcf(self, lag = 25, ax = None, xlabel = 'Time'):
        acfValues = self.getAcfUpToLag(lag)
        if (self.name is None):
            title = 'ACF for ' + str(lag) + ' lags'
        else:
            title = 'ACF for ' + str(lag) + ' lags - ' + self.name
        self.plotAuto(acfValues, lag, ax, title, xlabel)

    def getPacfLag(self, lag, yLags):
        '''returns the acf at a given lag
           inputs:
           * desired lag
           * yLags: matrix of lagged vectors to regress on
        '''
        if lag == 0:
            return 1.0
        else:
            return OLSRegression(yLags[lag:, 1:lag+1], # indep variables: all values lagged backwards
                                 self.data[lag:].reshape((-1,1)), # dep variable: values lagged forward
                                 addConstant = True).beta_hat[-1] # return the last regression coefficient

    def getPacfUpToLag(self, lag):
        ''' returns the PACF values up to the given lag '''
        pacf = []
        yLags = self.lagMatrix(lag)
        for i in range(lag):
            pacf.append(self.getPacfLag(i, yLags))
        return pacf

    def plotPacf(self, lag = 25, ax = None, xlabel = 'Time'):
        acfValues = self.getPacfUpToLag(lag)
        if (self.name is None):
            title = 'PACF for ' + str(lag) + ' lags'
        else:
            title = 'PACF for ' + str(lag) + ' lags - ' + self.name
        self.plotAuto(acfValues, lag, ax, title, xlabel)

    def lagMatrix(self, nlags, fill_vals = np.nan):
        '''Creates a matrix of lags'''
        yLags = np.empty((self.data.shape[0], nlags + 1))
        yLags.fill(fill_vals)
        ### Include 0 lag
        yLags[:, 0] = self.data
        for lag in range(1, nlags + 1):
            yLags[lag:, lag] = self.data[:-lag]
        return yLags

    def estimateAR(self, p, addConstant = True):
        ''' Fit an AR(p) on the Time Series. Returns the corresponding OLSRegression object'''
        yLags = self.lagMatrix(p)
        estimate = OLSRegression(yLags[p:, 1:p + 1], # indep variables: all values lagged backwards
                                 self.data[p:].reshape((-1, 1)), # dep variable: values lagged forward
                                 addConstant=addConstant)
        return estimate

    def augmentedDickeyFuller(self, lag):
        ''' Returns the statistic value of augmented Dickey-Fuller test
            (with given number of lags) for unit root '''
        lagMat = self.lagMatrix(lag)
        adfRegr = OLSRegression(lagMat[lag:, 1:lag + 1], # indep variables: all values lagged backwards
                                lagMat[lag:, 0] - lagMat[lag:, 1], # dep variable: first difference
                                addConstant=True)
        adfRegr.computeCovMatrix()
        return adfRegr.beta_hat[1] / adfRegr.heteroskedasticCovMatrix[1,1]

if __name__ == '__main__':
    # q1 = ARMA(ar = [0.8], ma = [0.7])
    # T = 1000
    # simTimeSeries = TimeSeries(q1.simulate(length = T))
    # # pdb.set_trace()
    # ar = simTimeSeries.estimateAR(1, addConstant = False)
    # ar.computeCovMatrix(heter = False)
    # ts_q1 = TimeSeries(ar.resid)
    # ts_q1.plotAcf()
    # print(ar.resid.shape)
    0

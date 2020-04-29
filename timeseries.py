import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from olsregr import OLSRegression
from arma import ARMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller
class TimeSeries:
    '''
    A class to perform early exploration on time series and fit some models
    Attributes:
    *** data: a numpy array representing a time series
    *** name (optional): name of the time series
    *** time_ticks (optional): the corresponding time labels
    '''
    def __init__(self, data, name=None, time_ticks=None):
        '''data: a numpy array representing a time series'''
        self.data = data
        self.name = name
        self.time_ticks = time_ticks

    def plot_diff_data(self, ax=None, num_diff=0):
        ''' plots the data differentiated diff (ie after applying
            delta ^ diff operator) '''
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        to_plot = self.__diff__(num_diff=num_diff)
        # difference the data diff times
        if self.time_ticks is not None:
            ax.plot(self.time_ticks[num_diff:], to_plot)
        else:
            ax.plot(to_plot)
        if self.name is not None:
            ax.set_title(str(self.name) + ' differenced ' + str(num_diff) + ' times ')
        else:
            ax.set_title('time series differenced ' + str(num_diff) + ' times ')

    def find_order(self, num_diff=3, lag=25):
        ''' plots ACF, PACF and data for several lags, with the objective of
            finding the order of integration (eg number of unit roots in ARIMA
            process) '''
        _, ax = plt.subplots(nrows=num_diff, ncols=3, figsize=(18, 5 * num_diff), squeeze=False)
        adfs = {}
        for i in range(num_diff):
            self.plot_diff_data(ax=ax[i][0], num_diff=i)
            self.plotAcf(lag=lag, ax=ax[i][1], num_diff=i)
            self.plotPacf(lag=lag, ax=ax[i][2], num_diff=i)
            dict_adf = self.adf_test(num_diff=i)
            adfs['diff ' + str(i)] = pd.Series(dict_adf)
        summar = pd.DataFrame(adfs)
        display(summar)

    def adf_test(self, lags=[3, 7, 10], num_diff=0):
        ''' Returns the statistic value of augmented Dickey-Fuller test
        (with given number of lags) for unit root '''
        dict_for_diff = {}
        ################## FOR DOCUMENTATION #####################
        # My code to compute the adf statistic
        # for lag in lags:
        #     lagMat = self.lagMatrix(lag + 1, num_diff=num_diff)
        #     # indep variables: previous level and all values lagged backwards
        #     indep = np.hstack([lagMat[lag:, 1].reshape((-1, 1)), lagMat[lag:, 1:lag] - lagMat[lag:, 2:lag + 1]])
        #     adfRegr = OLSRegression(X=indep,
        #                             Y=lagMat[lag:, 0] - lagMat[lag:, 1], # dep var: first difference
        #                             addConstant=True)
        #     adfRegr.__computeCovMatrix__()
        #     dict_for_diff['adf stat lag ' + str(lag)] = (adfRegr.beta_hat[1]
        #                     / np.sqrt(adfRegr.heteroskedasticCovMatrix[1,1]))
        # return dict_for_diff
        #############################################################
        diffed = self.__diff__(num_diff)
        for lag in lags:
            p_value = adfuller(diffed, maxlag=lag, regression='c')[1]
            dict_for_diff['adf pval, ' + str(lag) + ' lags'] = p_value
        return dict_for_diff

    def plotAuto(self, values, lag, title, ax=None, xlabel='Time'):
        '''plot function used for acf and pacf'''
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(0.4 * lag, 5))
        ax.plot(range(lag), values, marker='o', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-1.1, 1.1)
        # add vertical lines for readability
        for xc, yc in zip(range(lag), values):
            ax.plot([xc, xc], [0, yc], 'k-', lw=1)

    def plotAcf(self, lag=25, ax=None, xlabel='Time', num_diff=0):
        acfValues = self.getAcfUpToLag(lag, num_diff)
        if (self.name is None):
            title = 'ACF for ' + str(lag) + ' lags'
        else:
            title = 'ACF for ' + str(lag) + ' lags - ' + self.name
        self.plotAuto(values=acfValues, lag=lag, title=title, ax=ax, xlabel=xlabel)

    def getAcfLag(self, lag, num_diff=0):
        ''' returns the lag-th ACF of the data.
            NB: This assumes stationarity (ACF depends only on the lag)
            Gives the possibility to apply num_diff difference operator '''
        diffed = self.__diff__(num_diff)
        if lag == 0:
            return 1.0
        return (OLSRegression(diffed[:-lag].reshape((-1, 1)),
                diffed[lag:].reshape((-1, 1)),
                addConstant=True).beta_hat[-1])

    def getAcfUpToLag(self, lag, num_diff=0):
        ''' returns all ACF values up to the given lag '''
        acf = []
        for i in range(lag):
            acf.append(self.getAcfLag(lag=i, num_diff=num_diff))
        return acf

    def getPacfLag(self, lag, yLags):
        '''returns the acf at a given lag
           inputs:
           * desired lag
           * yLags: matrix of lagged vectors to regress on
        '''
        if lag == 0:
            return 1.0
        return OLSRegression(yLags[lag:, 1:lag+1], # indep variables: all values lagged backwards
                            yLags[lag:, 0], # dep variable: values lagged forward
                            addConstant=True).beta_hat[-1] # return the last regression coefficient

    def getPacfUpToLag(self, lag, num_diff=0):
        ''' returns the PACF values up to the given lag '''
        pacf = []
        diffed_lagged = self.lagMatrix(lag, num_diff=num_diff)
        for i in range(lag):
            pacf.append(self.getPacfLag(i, diffed_lagged))
        return pacf

    def plotPacf(self, lag=25, ax=None, xlabel='Time', num_diff=0):
        acfValues = self.getPacfUpToLag(lag, num_diff)
        if self.name is None:
            title = 'PACF for ' + str(lag) + ' lags'
        else:
            title = 'PACF for ' + str(lag) + ' lags - ' + self.name
        if num_diff:
            title = title + ' (' + str(num_diff) + 'd ifferences)'
        self.plotAuto(values=acfValues, lag=lag, title=title, ax=ax, xlabel=xlabel)

    def lagMatrix(self, nlags, fill_vals=np.nan, num_diff=0):
        '''Creates a matrix of lags'''
        diffed = self.__diff__(num_diff)
        yLags = np.empty((diffed.shape[0], nlags + 1))
        yLags.fill(fill_vals)
        ### Include 0 lag
        yLags[:, 0] = diffed
        for lag in range(1, nlags + 1):
            yLags[lag:, lag] = diffed[:-lag]
        return yLags

    def estimateAR(self, p, addConstant=True):
        ''' Fit an AR(p) on the Time Series. Returns the corresponding OLSRegression object'''
        yLags = self.lagMatrix(p)
        estimate = OLSRegression(yLags[p:, 1:p + 1], # indep variables: all values lagged backwards
                                 self.data[p:].reshape((-1, 1)), # dep variable: values lagged forward
                                 addConstant=addConstant)
        estimate.summary() # print estimation summary
        # return the corresponding AR object
        if (addConstant):
            return ARMA(ar=estimate.beta_hat[1:], constant=estimate.beta_hat[0])
        return ARMA(ar=estimate.beta_hat)

    def __diff__(self, num_diff):
        ''' applies the diff lag operator num_diff times (ie delta ^ num_diff) '''
        to_diff = np.copy(self.data)
        for d in list(range(num_diff)):
            to_diff = to_diff[1:] - to_diff[:-1]
        return to_diff

if __name__ == '__main__':
    # pdb.set_trace()
    # # Load data
    # data = pd.read_excel("Tbill10yr.xls",skiprows=14)
    #
    # # set index and change names
    # data.columns = ["Date", 'Yield']
    # test = TimeSeries(data=data['Yield'].values, time_ticks=data['Date'].values)
    # test.adf_test(num_diff=1)
    # test.find_order()
    0

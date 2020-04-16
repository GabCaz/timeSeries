import statsmodels
import numpy as np
import pixiedust
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from timeseriesutils import getDimensions
class OLSRegression:
    '''
    Attributes:
    *** X: in-sample regressor
    *** Y: in-sample sample values
    *** XprimXInv: (X'X)-1
    *** beta_hat: coefficients, computed in-sample
    *** resid: the model residuals
    *** homoskedasticCovMatrix: CovMatrix assuming homoskedasticity
    *** heteroskedasticCovMatrix: CovMatrix assuming heteroskedasticity
    *** heteroAutoCovMatrix: CovMatrix assuming heteroskedasticity and autocorrelation
    *** n: number of observations
    *** k: number of regressors
    '''
    def __init__(self, X, Y, conditionBound = 100000, addConstant = False):
        # Computing OLS point Estimate
        self.hasConstant = addConstant
        if addConstant:
            X = np.c_[np.ones((Y.shape[0],1)), X]
        self.X = X
        self.Y = Y
        XPrimX = np.dot(np.transpose(X), X)
        if np.linalg.cond(XPrimX) > conditionBound:
            print('Warning: Sample equivalent of 2nd moment matrix close to singular. Check multicollinearity')
        self.XprimXInv = np.linalg.inv(XPrimX)
        self.beta_hat = self.XprimXInv.dot(X.T).dot(Y).squeeze()

    def summary(self, alpha = 0.95):
        ''' prints coefficients, standard errors and common regression statistics '''
        self.computeCovMatrix()
        standardErrorsHom = [np.sqrt(se) for se in np.diagonal(self.homoskedasticCovMatrix)]
        standardErrorsWhite = [np.sqrt(se) for se in np.diagonal(self.heteroskedasticCovMatrix)]
        summaryTable = dict()
        for i, coefficient, seHom, seHet in zip(range(self.k), self.beta_hat, standardErrorsHom, standardErrorsWhite):
            summaryTable['beta ' + str(i)] = pd.Series(data = [coefficient, seHom, seHet],
                                                       index = ['point estimate', 'se (homoskedastic)',
                                                               'se (White estimator)'])
        display(pd.DataFrame(summaryTable).transpose())
        self.getRegressionStatistics()

    def computeCovMatrix(self):
        # Computing resid
        self.resid = self.Y.squeeze() - np.dot(self.X, self.beta_hat)
        # Getting covariance matrix in homoskedastic case
        self.n, self.k = getDimensions(self.X)
        self.homoskedasticCovMatrix = self.XprimXInv * (np.sum((self.resid)**2) / (self.n - self.k))
        # Getting covariance matrix is heteroskedastic case
        S = np.zeros((self.k, self.k)) # to compute central term in White Estimator
        for i in range(self.n): # computing White estimator
            S = S + np.outer(self.X[i,], self.X[i,]) * (self.resid[i] ** 2)
        self.heteroskedasticCovMatrix = self.XprimXInv.dot(S).dot(self.XprimXInv)

    def getRegressionStatistics(self):
        summaryTable = {'r2':[self.r2()], 'adjused r2:':[self.adjR2()]}
        display(pd.DataFrame(summaryTable))

    def r2(self):
        SSE = np.sum(self.resid ** 2)
        mean = np.mean(self.Y)
        SST = np.sum((self.Y - mean)**2)
        return 1 - SSE / SST

    def adjR2(self):
        return 1 - (1 - self.r2()) * (self.n - 1) / (self.n - self.k - 1)

import statsmodels
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from timeseriesutils import getDimensions
class OLSRegression:
    '''
    Attributes:
    *** X: in-sample regressors
    *** Y: in-sample values
    *** XprimXInv: (X'X)-1
    *** beta_hat: coefficients, computed in-sample
    *** resid: the model residuals
    *** homoskedasticCovMatrix: CovMatrix assuming homoskedasticity
    *** heteroskedasticCovMatrix: CovMatrix assuming heteroskedasticity
    *** heteroAutoCovMatrix: CovMatrix assuming heteroskedasticity and autocorrelation
    *** n: number of observations
    *** k: number of regressors
    '''
    def __init__(self, X, Y, conditionBound=100000, addConstant=False, warning=False):
        # Computing OLS point Estimate
        self.hasConstant = addConstant
        if addConstant:
            X = np.c_[np.ones((Y.shape[0],1)), X]
        self.X = X.squeeze()
        self.Y = Y.squeeze()
        XPrimX = np.dot(np.transpose(X), X)
        if np.linalg.cond(XPrimX) > conditionBound and warning:
            print('Warning: Sample equivalent of 2nd moment matrix close to singular. Check multicollinearity')
        self.XprimXInv = np.linalg.inv(XPrimX)
        self.beta_hat = self.XprimXInv.dot(X.T).dot(Y).squeeze()
        self.n, self.k = getDimensions(X)

    def summary(self, alpha=0.95):
        ''' prints coefficients, standard errors and common regression statistics '''
        self.__computeCovMatrix__()
        self.__computeCovMatrix__(white=False)
        self.__newey_west__()
        standardErrorsHom = [np.sqrt(se) for se in np.diagonal(self.homoskedasticCovMatrix)]
        standardErrorsWhite = [np.sqrt(se) for se in np.diagonal(self.heteroskedasticCovMatrix)]
        standardErrorsNw = [np.sqrt(se) for se in np.diagonal(self.newey_cov_matrix)]
        summaryTable = dict()
        for i, coefficient, seHom, seHet, seNw in zip(range(self.k), self.beta_hat,
                                                standardErrorsHom, standardErrorsWhite,
                                                standardErrorsNw):
            summaryTable['beta ' + str(i)] = pd.Series(data = [coefficient, seHom, seHet, seNw],
                                                       index = ['point estimate', 'se (homoskedastic)',
                                                               'se (White estimator)',
                                                               'se Newey-West'])
        display(pd.DataFrame(summaryTable).transpose())
        self.getRegressionStatistics()

    def __computeCovMatrix__(self, white=True):
        # Computing resid
        self.resid = self.Y.squeeze() - np.dot(self.X, self.beta_hat)
        if white:
            # Getting covariance matrix is heteroskedastic case
            x_eps = np.multiply(self.X, self.resid.reshape((-1,1)))
            sandwich = np.matmul(x_eps.T, x_eps)
            self.heteroskedasticCovMatrix = self.XprimXInv.dot(sandwich).dot(self.XprimXInv)
        else:
            # Getting covariance matrix in homoskedastic case
            self.homoskedasticCovMatrix = self.XprimXInv * (np.sum((self.resid)**2) / (self.n - self.k))

    def __newey_west__(self, lags=5):
        ''' compute the newey_west matrix with a given number of lag. Use if
        think errors are serially correlated '''
        # White Estimator
        self.resid = self.Y.squeeze() - np.dot(self.X, self.beta_hat)
        x_eps = np.multiply(self.X, self.resid.reshape((-1,1)))
        sandwich = np.matmul(x_eps.T, x_eps)
        # Add lags
        for lag in range(1, lags):
            # Credit to Nick Sanders (Berkeley) here
            x_lag = self.X[:-lag,]
            x_present = self.X[lag:,]
            eps_present = self.resid[lag:].reshape((-1,1))
            eps_lag = self.resid[:-lag].reshape((-1,1))
            present = np.multiply(x_present, eps_present)
            lagged = np.multiply(x_lag, eps_lag)
            sandwich += ((1 - lag) / (1 + lags)) * (np.matmul(present.T,lagged) + np.matmul(lagged.T, present))
        self.newey_cov_matrix = self.XprimXInv.dot(sandwich).dot(self.XprimXInv)



#         Xlag = X[0:T-lags]
#         Xpresent = X[lags:T]
#         ϵLag=ϵ[0:T-lags].A
#         Xpϵ=np.matrix(Xpresent.A*ϵPresent)
#         Xlϵ=np.matrix(Xlag.A*ϵLag)
#
#         sandwich = sandwich+(1-lags/(Lags+1))*(Xpϵ.T@Xlϵ+Xlϵ.T@Xpϵ)
#         #new=(1-lags/(Lags+1))*(Xpϵ.T@Xlϵ+Xlϵ.T@Xpϵ)
#         #print(new)
#
#
#     var_β = XprimeX.I@sandwich@XprimeX.I
#     return var_β

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

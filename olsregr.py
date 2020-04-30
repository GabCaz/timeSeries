from scipy.stats import norm # for confidence intervals
import numpy as np
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
    def __init__(self, X, Y, conditionBound=100000, addConstant=False, warning=False, Z=None):
        ''' creating a regression object, possiblly with instrument variable Z '''
        # Computing OLS point Estimate
        if Z is None:
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
        if Z is not None:
            if addConstant:
                X = np.c_[np.ones((Y.shape[0],1)), X]
                Z = np.c_[np.ones((Y.shape[0],1)), Z]
            ZPrimX = np.dot(np.transpose(Z), X)
            if np.linalg.cond(ZPrimX) > conditionBound and warning:
                print('Warning: Sample equivalent of 2nd moment matrix close to singular. Check multicollinearity')
            self.ZPrimXInv = np.linalg.inv(ZPrimX)
            self.beta_hat = self.ZPrimXInv.dot(Z.T).dot(Y).squeeze()
        self.n, self.k = getDimensions(X)

    def summary(self):
        ''' prints coefficients, standard errors and common regression statistics '''
        self.__computeCovMatrix__()
        self.__computeCovMatrix__(white=False)
        self.__newey_west__()
        standardErrorsHom = [np.sqrt(se) for se in np.diagonal(self.homoskedasticCovMatrix)]
        standardErrorsWhite = [np.sqrt(se) for se in np.diagonal(self.heteroskedasticCovMatrix)]
        standardErrorsNw = [np.sqrt(se) for se in np.diagonal(self.newey_cov_matrix)]
        tstats = self.beta_hat / standardErrorsWhite
        pvalues = (1 - stats.norm.cdf(np.abs(tstats))) * 2
        summaryTable = dict()
        for i, coefficient, seHom, seHet, seNw, pval in zip(range(self.k), self.beta_hat,
                                                standardErrorsHom, standardErrorsWhite,
                                                standardErrorsNw, pvalues):
            summaryTable['beta ' + str(i)] = pd.Series(data=[coefficient, seHom, seHet, seNw, pval],
                                                       index=['point estimate', 'se (homoskedastic)',
                                                               'se (White estimator)',
                                                               'se Newey-West', 'p-value (using White)'])
        display(pd.DataFrame(summaryTable).transpose())
        self.getRegressionStatistics()

    def __computeCovMatrix__(self, white=True):
        # Computing resid
        self.resid = self.Y.squeeze() - np.dot(self.X, self.beta_hat)
        if white:
            # Getting covariance matrix is heteroskedastic case
            x_eps = np.multiply(self.X, self.resid.reshape((-1, 1)))
            sandwich = np.matmul(x_eps.T, x_eps)
            self.heteroskedasticCovMatrix = self.XprimXInv.dot(sandwich).dot(self.XprimXInv)
        else:
            # Getting covariance matrix in homoskedastic case
            self.homoskedasticCovMatrix = self.XprimXInv * (np.sum((self.resid)**2) / (self.n - self.k))

    def __newey_west__(self, lags=5):
        ''' compute the newey_west matrix with a given number of lag. Use if
        think errors are serially correlated '''
        self.resid = self.Y.squeeze() - np.dot(self.X, self.beta_hat)
        x_eps = np.multiply(self.X, self.resid.reshape((-1, 1)))
        sandwich = np.matmul(x_eps.T, x_eps) # White Estimator
        # Add lags
        for lag in range(1, lags):
            # Nick Sanders's implementation idea here; mistakes are mine
            x_lag = self.X[:-lag,]
            x_present = self.X[lag:,]
            eps_present = self.resid[lag:].reshape((-1, 1))
            eps_lag = self.resid[:-lag].reshape((-1, 1))
            present = np.multiply(x_present, eps_present)
            lagged = np.multiply(x_lag, eps_lag)
            sandwich += ((1 - lag / (1 + lags)) * (np.matmul(present.T, lagged)
                        + np.matmul(lagged.T, present)))
        self.newey_cov_matrix = self.XprimXInv.dot(sandwich).dot(self.XprimXInv)

    def getRegressionStatistics(self):
        SSE = np.sum(self.resid ** 2)
        mean = np.mean(self.Y)
        SST = np.sum((self.Y - mean)**2)
        r2 = 1 - SSE / SST
        adjr2 = 1 - (1 - r2) * (self.n - 1) / (self.n - self.k - 1)
        logL = - (self.n / 2) * np.log(2 * np.pi * ((SSE / (self.n - self.k)))) - (self.n - self.k) / 2
        aic = -2 * logL + 2 * self.k
        bic = -2 * logL + 2 * self.k * np.log(self.n)
        hannan_quinn = -2 * logL + 2 * self.k * np.log(np.log(self.n))
        summaryTable = {'r2':[r2], 'adjused r2':[adjr2], 'aic':aic, 'bic':bic,
                        'Hannan Quinn':hannan_quinn}
        display(pd.DataFrame(summaryTable))

    def significance(self, cov_matrix='ols'):
        ''' returns t-stats and p-values for each coefficient '''
        if cov_matrix == 'ols':
            # standard errors are ols
            self.__computeCovMatrix__(white=False)
            ses = [np.sqrt(se) for se in np.diagonal(self.homoskedasticCovMatrix)]
        elif cov_matrix == 'white':
            self.__computeCovMatrix__(white=True)
            ses = [np.sqrt(se) for se in np.diagonal(self.heteroskedasticCovMatrix)]
        elif cov_matrix == 'nw':
            self.__newey_west__()
            ses = [np.sqrt(se) for se in np.diagonal(self.newey_cov_matrix)]
        tstats = self.beta_hat / ses
        pvalues = (1 - stats.norm.cdf(np.abs(tstats))) * 2
        return ses, tstats, pvalues

    def confidence_interval(self, cov_matrix='ols', alpha=0.95, disp=True):
        ''' prints the coefficients, the p-value for the test 'is zero', and an
        alpha confidence interval. Depending on the method, use ols, white or
        Newey-West variance-covariance matrix '''
        ses, tstats, pvalues = self.significance(cov_matrix)
        summaryTable = dict()
        tstats = self.beta_hat / ses
        pvalues = (1 - stats.norm.cdf(np.abs(tstats))) * 2
        conf_multiplicator = norm.ppf((1 + alpha) / 2)
        for i, coeff, se, pval in zip(range(self.k), self.beta_hat, ses, pvalues):
            interv = (str(np.round(coeff - conf_multiplicator * se, 2)) + ' - '
                        + str(np.round(coeff + conf_multiplicator * se, 2)))
            summaryTable['beta ' + str(i)] = pd.Series(data=[coeff, se, pval, interv],
                                                       index=['point estimate',
                                                       'se using ' + cov_matrix,
                                                       'p-value (using ' + cov_matrix + ')',
                                                        str(alpha) + ' confidence interval'])
        if disp:
            display(pd.DataFrame(summaryTable).transpose())

if __name__ == '__main__':
    # import pdb
    # from statsmodels.regression.linear_model import OLS
    # from pandas_datareader.famafrench import get_available_datasets
    # import pandas_datareader.data as web
    # import datetime
    # from statsmodels.tools.tools import add_constant
    # pdb.set_trace()
    # start = datetime.datetime(1963, 7, 1)
    # data_5factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start)
    # data_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=start)
    # data_25port = web.DataReader('25_Portfolios_ME_Prior_12_2', 'famafrench', start=start)
    # df_5factors = data_5factors[0]
    # df_mom = data_mom[0]
    # df_25port = data_25port[0]
    # dep_var = df_25port['SMALL LoPRIOR'].values
    # indep_var = df_5factors['Mkt-RF'].values
    # my_reg = OLSRegression(indep_var, dep_var, addConstant=True)
    # sm_reg = OLS(dep_var, add_constant(indep_var)).fit()
    # my_reg.__newey_west__()
    # np.sqrt(NeweyWest(dep_var, add_constant(indep_var), my_reg.beta_hat, 5))
    0

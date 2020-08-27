''' Useful methods and routines (built on top of statsmodesl) for time series
    analysis. The script has two parts:
    1. Plotting and early analysis methods (using matplotlib, etc.)
    2. Routines built on top of statsmodels to perform common operations within
       one call only (e.g. check for cointegration, test common OLS
       assumptions...), so that the very essential information is readily
       presented for such common tasks '''
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels import tsa
from olsregr import OLSRegression
from timeseries import TimeSeries
from scipy.stats import chi2 # for chi-square test
import statsmodels.stats.diagnostic as diagnostic
from statsmodels.regression.rolling import RollingOLS
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

def plotAllDistribs(dictOfDistribs, add_normal=True):
    '''
    Purpose: given a dictionary of array of numbers, will plot the distribution
    of each array
    '''
    plt.figure(figsize=[10, 6 * len(dictOfDistribs)]) # make large figure
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

def rolling_coefficients(X, Y, window_size=60, addConstant=True):
    ''' returns coefficients of regression windows of Y on windows of X,
    with given window_sise.
    Returns: a list of arrays, where each array is the coefficient for a window '''
    rolling_coeffs = [] # list of coefficients we will return
    # for each window of observations...
    for i in range(X.shape[0] - 60):
        window_X = X[i:i + 60] # carve out windows
        window_Y = Y[i:i + 60]
        reg_window = OLSRegression(X=window_X, Y=window_Y, addConstant=True)
        rolling_coeffs.append(reg_window.beta_hat)
    return rolling_coeffs

# Source: https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
def abline(slope, intercept, ax=None, label=None):
    """Plot a line from slope and intercept"""
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    if label is None:
        ax.plot(x_vals.squeeze(), y_vals.squeeze(), '--')
    else:
        ax.plot(x_vals.squeeze(), y_vals.squeeze(), '--', label=label)
        ax.legend()

def plotline(point1, point2, ax=None, label=None):
    ''' given two points on a line (two tupples), plots line '''
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - x1 * slope
    abline(slope, intercept, ax, label)

def my_grs_test(regr_vector, market_sr):
    ''' given the vector of regressions and market Sharpe-Ratio, returns test statistic
        and p-value to jointly test that all the intercepts are 0 '''
    alphas = np.array([reg.beta_hat[0] for reg in regr_vector]).reshape((1, -1)) # array with all the regression constants
    resids = np.array([reg.resid for reg in regr_vector]).T # collecting all residuals
    cov_resid = np.dot(resids.T, resids) / resids.shape[0] # covariance matrix of residuals
    cov_resid_inv = np.linalg.inv(cov_resid)
    grs_test_val = (resids.shape[0] * alphas.dot(cov_resid_inv).dot(alphas.T).squeeze()
            / (1 + market_sr ** 2))
    p_val = 1 - chi2.cdf(grs_test_val, len(regr_vector)) # p-value of test 'GRS test statistics is 0'
    return grs_test_val, p_val

### Below this point are routines on top of statsmodels
def test_multicollinearity(ols_fit):
    ''' given a SM fitted OLS object, says whether risk of multicollinearity '''
    return np.linalg.cond(ols_fit.model.exog)

def test_hetro(ols_fit, endog=True):
    ''' given SM fitted OLS object, helps determine whether residuals are heteroskedastic
        or homoskedastic  '''
    squared_resid = ols_fit.resid**2
    TimeSeries(data=squared_resid, name='Squared regression residuals').find_order(1)
    if not endog:
        summary = {'ARCH test (5 lags)':diagnostic.het_arch(ols_fit.resid, nlags=5)[1]}
    else:
        summary = {'ARCH test (5 lags)':[diagnostic.het_arch(ols_fit.resid, nlags=5)[1]],
                   'Breusch-Pagan test':[diagnostic.het_breuschpagan(ols_fit.resid, ols_fit.model.exog)[1]]}
    display(pd.DataFrame(summary, index=['p values']))

def test_autocor(ols_fit, nlags=5):
    ''' test for autocorrelation in residuals '''
    resid = ols_fit.resid
    TimeSeries(data=resid, name='Regression residuals').find_order(1)
    pv = diagnostic.acorr_breusch_godfrey(ols_fit, nlags=nlags)[1]
    display(pd.DataFrame({'Breusch Godefrey test for autocorr. ' + str(nlags) + ' lags':pv},
                         index=['p value']))

def test_normality(ols_fit):
    from statsmodels.graphics.gofplots import qqplot
    from statsmodels.stats.stattools import jarque_bera
    plotAllDistribs({'residuals':ols_fit.resid})
    qqplot(ols_fit.resid)
    display(pd.DataFrame({'jarque bera for normality':jarque_bera(ols_fit.resid)[1]},
                         index=['p value']))

def test_stable(ols_fit, window=60):
    ''' Early exploration to see whether ols model has stable coefficients '''
    roll = RollingOLS(ols_fit.model.endog, ols_fit.model.exog, window=window).fit()
    fig, ax = plt.subplots(1, 1, figsize=(15,8))
    for i in range(roll.params.shape[1]):
        b = roll.params[:,i]
        plt.plot(range(len(ols_fit.model.exog)), b, label='b'+str(i))
        se = roll.bse[:,i]
        plt.fill_between(range(len(ols_fit.model.exog)), b-2*se, b+2*se, alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

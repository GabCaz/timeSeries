'''AMAN's'''
# Utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels import tsa
from scipy.stats import chi2 # for chi-square test
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

### "wrapped" GLS statsmodel routine : from lab 3
def _sm_calc_gls(y, x, Dtilde, addcon=True, cov_type=None, sig_level=.05, summary=0):
    """Wrapper for statsmodels GLS regression
       Note: we need to specify the "D" matrix in GLS.
    """
    if addcon:
        X = sm.add_constant(x)
    else:
        X = x
    ### SEs...
    if cov_type==None:
        gls_results = sm.regression.linear_model.GLS(y,X, sigma=Dtilde).fit(cov_type='nonrobust')
    else:
        gls_results = sm.regression.linear_model.GLS(y,X, sigma=Dtilde).fit(cov_type=cov_type)
    ### print out the OLS estimation results
    if summary==1:
        print(gls_results.summary())
    gls_beta_hat = gls_results.params # beta_hat
    gls_resids   = gls_results.resid  # resids
    gls_ci       = gls_results.conf_int(alpha=sig_level)[-2:] # 95% confidence intervals
    return gls_beta_hat, gls_resids, gls_ci

### "wrapped" OLS statsmodel routine: from lab 3
def _sm_calc_ols(y, x, addcon=True, cov_type=None, sig_level=.05, summary=0):
    """Wrapper for statsmodels OLS regression
    """
    if addcon:
        X = sm.add_constant(x)
    else:
        X = x
    if cov_type==None:
        ols_results = sm.OLS(y,X).fit(cove_type='nonrobust')
    else:
        ols_results = sm.OLS(y,X).fit(cov_type=cov_type)
    ### print out the OLS estimation results
    if summary==1:
        print(ols_results.summary())
    ols_beta_hat = ols_results.params # beta_hat
    ols_resids   = ols_results.resid  # resids
    ols_ci       = ols_results.conf_int(alpha=sig_level)[-2:] # 95% confidence intervals
    return ols_beta_hat, ols_resids, ols_ci

### From lab 3: computing skewness
def nth_moment(y, center, n):
    """ Calculates nth moment around 'center"""
    return np.sum((y - center)**n) / y.size

### From lab 4: use information criteria to compare different lag length for ARMA
### Let's use information criteria
def calc_arma_ic(x, order_list):
    ic_array = np.empty((3, len(order_list)))
    for idx, order in enumerate(order_list):
        # Set up model with given order
        model = tsa.arima_model.ARMA(x , order=order )
        # Fit and save criteria
        try:
            start_params = None
            res = model.fit(start_params=start_params, maxiter=500 )
        except Exception:
            start_params = np.zeros(np.sum(order)+1)
            res = model.fit(start_params=start_params, maxiter=500 )
        ic_array[0,idx] = res.aic
        ic_array[1,idx] = res.hqic
        ic_array[2,idx] = res.bic

    ### Return min orders for each criteria
    aic_order = order_list[ np.argmin(ic_array[0,:]) ]
    hqic_order = order_list[ np.argmin(ic_array[1,:]) ]
    bic_order = order_list[ np.argmin(ic_array[2,:]) ]

    ### Tuple of best orders
    ic_orders = (aic_order, hqic_order, bic_order)
    return ic_orders, ic_array

### From lab 4: creating ARMA from roots
def _roots2coef(roots):
    """Given roots, get the coefficients"""
    ### SymPy: package for symbolic computation
    from sympy import symbols, expand, factor, collect, simplify, Mul

    N_roots = len(roots)
    L = symbols("L", commutative=False) # symbolic variable
    ## Construct lag polynomial in the canonical form
    expr = expand(1)
    for r in roots:
        expr*= -(L - r)
    expr_expand = expand(expr)
    expr_expand = expand((expr_expand.as_coefficients_dict()[1]**-1)*expr_expand).evalf(3)

    ## factor out the lag polynomials and get "factor list" in the canonical form
    expr_factor = factor(expr_expand)
    for f in range(1, len(expr_factor.args)):
        if f==1:
            expr = expand(expr_factor.args[f]*-1).evalf(3)
        else:
            expr = Mul(expr, expand(expr_factor.args[f]*-1).evalf(3))

    coef_list = [expr_expand.coeff(L,n) for n in range(N_roots + 1)]
    ### convert to numpy floats
    coefs = np.array(coef_list).astype(float)
    ### normalize zero lag to 1
    coefs /= coefs[0]
    return coefs, expr, expr_expand

def arma_from_roots(ar_roots=[], ma_roots=[]):
    """Create an ARMA model class from roots"""
    ar_coef, ar_expr, ar_expr_expand = _roots2coef(ar_roots)
    if len(ma_roots)>0:
        ma_coef, ma_expr, ma_expr_expand = _roots2coef(ma_roots)
    print("AR lag polynomials in the form:", ar_expr_expand)
    if len(ma_roots)>0:
        print("MA lag polynomials in the form:", ma_expr_expand, "\n")
    print("factored AR lag polynomials in the form:", ar_expr)
    if len(ma_roots)>0:
        print("factored MA lag polynomials in the form:", ma_expr, "\n")
    if len(ma_roots)>0:
        arma_process = sm.tsa.ArmaProcess(ar_coef, ma_coef)
    else:
        arma_process = sm.tsa.ArmaProcess(ar_coef, [1])
    ### Note: arma_process' has many helpful methods: arcoefs, macoefs, generate_sample, ...
    return arma_process

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

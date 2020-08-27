from lib_lab import *
### "wrapped" GLS statsmodel routine : from lab 3
def _sm_calc_gls(y, x, Dtilde, addcon=True, cov_type=None, sig_level=.05, summary=0):
    """ Wrapper for statsmodels GLS regression
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
    return gls_results, gls_beta_hat, gls_resids, gls_ci

### "wrapped" OLS statsmodel routine: from lab 3
def _sm_calc_ols(y, x, addcon=True, cov_type=None, sig_level=.05, summary=1):
    """Wrapper for statsmodels OLS regression
    x and y must be array-like.
    cov_type can be:
    ** None (will default to nonrobust)
    ** ‘HC0’: heteroscedasticity robust covariance (i.e. White)
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
    return ols_results, ols_beta_hat, ols_resids, ols_ci

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
        model = tsa.arima_model.ARMA(x , order=order)
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

### From lab 4: create an AR process from roots
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

### From lab 4:  simple "brute force" IV regression function
def calc_iv(y, x, z, addcon=True):
    Nobs = y.shape[0]
    if addcon:
        X = np.c_[np.ones((Nobs, 1)), x] # append the [Nobs x 1] columns of ones.
    else:
        X = x
    k = X.shape[1]
    ZtX = np.dot(Z.T, X)
    Zty = np.dot(Z.T, y)
    beta_iv_hat =  np.linalg.solve(ZtX, Zty)
    return beta_iv_hat

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

### From Lab 4: use information criteria to select the right ARMA specification
def calc_arma_ic(x, order_list):
    ''' Input a list of possible order tuples (p, q), returns the best
        order for each information criteria  '''
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
    print("(AIC, BIC, Hannan-Quinn): (p,q) =", ic_orders)
    return ic_orders, ic_array

### From Lab 5: Perform ADF test on each time series in the data frame
# (Apply to df and cointegration error to test)
def ADF_test(df):
    for j in range(0, df.shape[1]):
        adf_result1 = sm.tsa.stattools.adfuller(df.iloc[:,j])
        print('\tY'+str(j)+': ADF statistic (p-value): %0.3f (%0.3f)' % (adf_result1[0], adf_result1[1]),
                      '\n\tcritical values', adf_result1[4],'\n')

### From lab 5: test cointegration for alpha
def get_cointegrating_vec(x1, x2, alpha, remove_trend=False):
    """
    Get cointegrating error "z" from alpha,
    and check for unit root
    """
    print('Cointegrating vector: ', alpha)
    X = np.c_[x1, x2]
    z = np.dot(X, alpha)
    if remove_trend:
        T = np.array(range(z.shape[0]))
        z = sm.OLS(z, sm.add_constant(T)).fit().resid
    ### Check for unit root
    adf_result = sm.tsa.stattools.adfuller(z)
    print('\nADF stat (p-val): %0.3f (%0.3f)' % (adf_result[0], adf_result[1]),
         '\n\tcritical values:', adf_result[4], '\n')
    return z

### From lab 5: comparing information criteria for differen VECM lag orders
def bic(n, k, ll):
    return round(np.log(n)*k-2*ll,3)
def aic(k, ll):
    return round(2*k-2*ll,3)
def get_info_vecm(df, f, t):
    p = dict()
    for i in range(f, t+1):
        vecm_model = VECM(np.c_[df['Y1'], df['Y2']], k_ar_diff=i).fit()
        vecm_model.summary()

        n  = vecm_model.nobs
        k  = len(vecm_model.stderr_params)
        ll = vecm_model.llf

        print("aic for lag"+str(i)+": "+str(aic(k, ll)))
        print("bic for lag"+str(i)+": "+str(bic(n, k, ll)))
        print()
        p[i] = bic(n, k, ll)
    pstar = min(p, key=p.get)
    return p, pstar

### From lab 7: home-made MA (moving average)
def get_ma(x, w):
    l = len(x)
    y = np.zeros(l)
    for i in range(l):
        if i < w:
            y[i] = np.mean(x[0:(i+1)])
        else:
            y[i] = np.mean(x[(i+1-w):(i+1)])
    return y

### From lab 7: wrapper for GARCH(p, q) usage
def _arch_model_est(y, y_RV, ARCHModel_dict, max_horizon, train_enddt=None, test_startdt=None, summarize=True,
                    plot_vol_measures=True, model_name='', annualize=True, sim_t = 1000, y_title = 'Return (%)', x_title ='Years',
                     y_RV_title = 'Realized Vol $(\sigma_{t+1})$', y_RV_label = '$RV_{t} =\sigma_{t+1}$'):
    """
    PARAM: "y" is 'DataFrame' with observed process for full sample
    PARAM: "y_RV" is the observed realized volatility of the observed process for the full sample.
    PARAM: "ARCHModel_dict" is a 'dict' with parameter values for arguments used to construct a model instance and
           other estimation arguments
    PARAM: "train_enddt" is 'datetime' with date of last observation used in training/estimation sample
    PARAM: "test_startdt" is 'datetime' with date of first observation (exclusively) used in testing/forecasting sample
    PARAM: "summarize" is a 'bool'=True if we want to display estimation summary, ='False' otherwise
           [default: summarize='True']
    PARAM: "plot_vol_measures" is a 'bool'=True if we want to plot the following:
            1. estimated conditional volatility,
            2. 1-step ahead forecasted conditional volatility
            3. realized volatility
            ='False' otherwise [default: plot_vol_measures=True]
           [default: plot_condvol='True']
    PARAM: "annualize" is a 'bool'=True if volatility measures (observed or estimated) are annualized, ='False' otherwise
           [default: plot_condvol='True']

    Examples of usage:
    *** ARCH(1)
    Params = pd.DataFrame(index=['mu', 'omega','omega1', 'alpha[1]', 'alpha[2]', 'alpha[3]', 'alpha[4]', 'gamma[1]', 'beta[1]'])
    arch_dict   = {'mean': 'Constant', 'lags': 0, 'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 0, 'dist': 'Normal',
                    'freq': 'M', 'disp': 'off'}
    train_enddt  = None
    test_startdt = df['Y1'].index.min()
    arch_dict1   =_arch_model_est(y=df['Y1'], y_RV=np.abs(df['Y1']), ARCHModel_dict=arch_dict,
                                    max_horizon=5, test_startdt=test_startdt, model_name='ARCH(1)', annualize=False)

    *** GARCH(1,1)
    garch11_dict = {'mean': 'Constant', 'lags': 0, 'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'Normal',
                'freq': 'M', 'disp': 'off'}
    train_enddt = None
    test_startdt = ixic_ret['ret'].index.min()
    garch11_IXIC_dict =_arch_model_est(y=ixic_ret['ret'], y_RV=ixic_ret['vol'], ARCHModel_dict=garch11_dict,
                                    max_horizon=5, test_startdt=test_startdt, model_name='GARCH(1,1)', annualize=False)
    """
    ### Training sample
    am = arch_model(y, mean=ARCHModel_dict['mean'], lags=ARCHModel_dict['lags'], vol=ARCHModel_dict['vol'],
                       p=ARCHModel_dict['p'], o=ARCHModel_dict['o'], q=ARCHModel_dict['q'],
                       dist=ARCHModel_dict['dist'])

    if train_enddt==None:
        am_est = am.fit(disp=ARCHModel_dict['disp'])
    else:
        am_est = am.fit(disp=ARCHModel_dict['disp'], last_obs=train_enddt)

    am_dict = ARCHModel_dict.copy()
    am_dict['bic'] = am_est.bic
    am_dict['log-likelihood'] = am_est.loglikelihood
    am_dict['params'] = am_est.params
    am_dict['cond_vol'] = am_est.conditional_volatility

    if summarize:
        print('\n**** '+y.name+' ****\n\n', am_est.summary(), 'n')

    ### h-step ahead forecasts
    if test_startdt==None:
        am_rv_forecasts_h = (am_est.forecast(horizon=max_horizon, align='target', method='simulation', simulations=100).variance)**0.5
    else:
        am_rv_forecasts_h = (am_est.forecast(horizon=max_horizon, align='target', start=test_startdt).variance)**0.5

    # simulate data
    am_dict['sim_data']   = am.simulate(am_est.params.values, sim_t)

    ### MSE, RMSE, & MAE
    MSE, RMSE, MAE = {}, {}, {}
    for h in range(1, max_horizon+1):
        error_h = (y[test_startdt] - am_rv_forecasts_h['h.'+str(h)]).dropna()
        T_h = len(error_h)
        MSE[h] = (1/T_h)*np.sum(error_h**2)
        RMSE[h] = ((1/T_h)*np.sum(error_h**2))**0.5
        MAE[h] = (1/T_h)*np.sum(np.abs(error_h))

    am_dict['max_forecast_h'] = max_horizon
    am_dict['rv_forecasts_h'] = am_rv_forecasts_h
    am_dict['MSE'] = MSE
    am_dict['RMSE'] = RMSE
    am_dict['MAE'] = MAE

    if annualize:
        a_label = '$\\bf{annualized}$'
        if ARCHModel_dict['freq']=='D':
            scale = 252**0.5
        elif ARCHModel_dict['freq']=='M':
            scale = 12**0.5
    else:
        a_label = '$\\bf{annualized}$'
        scale = 1

    if plot_vol_measures:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,8))
        axes.plot(scale*y_RV, lw=0.75, label=y_RV_label)
        axes.plot(scale*am_est.conditional_volatility, lw=1.5, ls='--',
                  label='$'+model_name+'$: $\widehat{\sigma}_{t+1}$')
        axes.plot(scale*am_rv_forecasts_h['h.1'], lw=1.5, alpha=.5,
                  label='$'+model_name+'$: $E_t[\widehat{\sigma}_{t+1|t}]$')
        axes.set_xlabel(x_title, fontsize=18)
        axes.set_ylabel(y_title, fontsize=18)
        #title_label = 'log '+y.name+' returns:\n'+a_label+'\nRealized Vol $(\sigma_{t+1})$,' +\
        #              '\nEstimated Conditional Vol $(\widehat{\sigma}_{t+1})$'+\
        #              '\n$Forecasted$ $(h='+str(1)+')$ Realized Vol $(\sigma_{t+h|t})$'
        title_label = 'log '+y.name+' returns:\n'+y_RV_title+\
                      '\nEstimated Conditional Vol $(\widehat{\sigma}_{t+1})$'+\
                      '\n$Forecasted$ $(h='+str(1)+')$ Realized Vol $(\sigma_{t+h|t})$'
        axes.set_title(title_label, fontsize=18)
        axes.legend(fontsize=16, loc='best')
        fig.tight_layout()
    return am_dict

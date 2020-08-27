from __future__ import print_function, division

import warnings
warnings.simplefilter("ignore")

import copy
from cycler import cycler
from dotenv import load_dotenv
from functools import partial
import itertools
from ipy_table import *
from IPython.display import display, HTML
import math
import numba
from numba import njit
import numpy as np
import os
import random
from sklearn.decomposition import PCA
from sympy import symbols, expand, factor, collect, simplify, Mul
from sympy import Symbol
from sympy.solvers import solve
import sys
from tabulate import tabulate
from time import time, sleep
import yfinance as yf
from yahoofinancials import YahooFinancials
### Import 'arch'
import arch
from arch import arch_model

# Import date-related
from calendar import month_abbr
from datetime import date, datetime, timedelta
import datetime as dt
import dateutil


### Matplotlib imports
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.dates
from matplotlib import colors as mcolors
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

### Matplotlib MCMC
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk


### Import Pandas-related
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import Day, MonthEnd

from distutils.version import StrictVersion
if StrictVersion(pd.__version__) >= StrictVersion('0.19'):
    import pandas_datareader.data as web
    from pandas_datareader import wb
else:
    import pandas.io.data as web
    from pandas.io import wb

import pandas_datareader as pdr


### Scipy imports
from scipy.stats import beta, chi2, t, norm
import scipy.linalg
from scipy.linalg import toeplitz
import scipy.optimize
from scipy.optimize import minimize
import scipy.signal
import scipy as sp
import scipy.special
import scipy.stats

from pykalman import KalmanFilter

### statsmodels imports
from statsmodels import tsa
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels import multivariate
from statsmodels import regression, stats
from statsmodels.sandbox.regression import gmm
from statsmodels.sandbox.regression.gmm import GMM
import statsmodels.stats.diagnostic as smd
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults
from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess, VARResults
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order


import wrds
import sqlalchemy as sa
from wrds import __version__ as wrds_version
from sys import version_info



##########
def multi_table(table_list):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    CSS = """.output {vertical-align: middle;}"""

    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
        '<style>{}</style>'.format(CSS)
    )


def set_stuff():
	mpl.rcParams['lines.linewidth'] = 1.5
	mpl.rcParams['lines.color'] = 'blue'
	mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b'])
	mpl.rcParams['legend.fancybox'] = True
	mpl.rcParams['legend.fontsize'] = 14
	mpl.rcParams['axes.facecolor'] = '#f0f0f0'
	mpl.rcParams['axes.labelsize'] = 15
	mpl.rcParams['axes.axisbelow'] = True
	mpl.rcParams['axes.linewidth'] = 1.2
	mpl.rcParams['axes.labelpad'] = 0.0
	mpl.rcParams['axes.xmargin'] = 0.05  # x margin.  See `axes.Axes.margins`
	mpl.rcParams['axes.ymargin'] = 0.05  # y margin See `axes.Axes.margins`
	mpl.rcParams['xtick.labelsize'] = 18
	mpl.rcParams['ytick.labelsize'] = 18
	mpl.rcParams['figure.subplot.left'] = 0.08
	mpl.rcParams['figure.subplot.right'] = 0.95
	mpl.rcParams['figure.subplot.bottom'] = 0.07

	### set display precision when using pandas dataframes
	np.set_printoptions(precision=3)
	pd.set_option('precision', 3)
	pd.set_option('display.float_format', lambda x: '%.3f' % x)

	plt.rc('text', usetex=False)
	plt.style.use('ggplot')

	fsize = (10,7.5) # figure size
	tsize = 18 # title font size
	lsize = 16 # legend font size
	csize = 14 # comment font size
	grid = True # grid
	display(HTML("<style>.container { width:100% !important; }</style>"))

def win2(df, v, by, w1, w2):
    def q1(x):
        return x.quantile(w1)

    def q2(x):
        return x.quantile(w2)

    x    =  df.groupby([by]).agg({v: [q1, q2]})
    x.reset_index(inplace = True)
    x.columns = [by, 'w_1', 'w_2']
    df   = pd.merge(df, x, on = by, how = 'outer')

    df.loc[df[v] < df['w_1'], v] = df.loc[df[v] < df['w_1'], 'w_1']
    df.loc[df[v] > df['w_2'], v] = df.loc[df[v] < df['w_2'], 'w_2']

    #df[v][df[v] < df['w_1']]  = df['w_1'][df[v] < df['w_1']]
    #df[v][df[v] > df['w_2']]  = df['w_2'][df[v] < df['w_2']]
    df   = df.drop(columns=['w_1', 'w_2'])
    return(df)

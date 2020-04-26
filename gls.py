from timeseriesutils import getDimensions
import numpy as np
import scipy.stats as stats
import pandas as pd
class GLS:
    ''' a class for gls
        Attributes:
        *** X: in-sample regressors
        *** Y: in-sample values
        *** D: the D matrix, specifying the covariance matrix of residuals '''
    def __init__(self, X, Y, D, addConstant=False):
        if addConstant:
            X = np.c_[np.ones((Y.shape[0],1)), X]
        self.X = X.squeeze()
        self.Y = Y.squeeze()
        self.n, self.k = getDimensions(X)
        XprimDinv = X.T.dot(np.linalg.inv(D))
        XprimDinvX_inv = np.linalg.inv(XprimDinv.dot(X))
        XprimDinvY = XprimDinv.dot(Y)
        self.beta_hat = XprimDinvX_inv.dot(XprimDinvY)
        y_hat = np.dot(X, self.beta_hat)
        self.resid = Y.squeeze() - y_hat.squeeze()

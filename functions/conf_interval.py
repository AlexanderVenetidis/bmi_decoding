"""
to compute confidence interval based on t or normal distribution
"""

import numpy as np
import scipy.stats as st

def t_confint(x,ci=0.95):
    """
    to compute confidence interval based on t-distribution

    Parameters
    ----------
    x : ndarray
        input data.
    ci : float
        confidence level, ci = 0.95 for 95% confidence interval.

    Returns
    -------
    x_mean : ndarray
        mean value.
    lower : ndarray
        lower bound value.
    upper : ndarray
        upper bound value.

    """
    x_mean = np.mean(x,axis=0)
    x_sem = st.sem(x,axis=0)
    lower, upper = st.t.interval(ci, x.shape[0]-1, loc=x_mean, scale=x_sem)        
    return x_mean, lower, upper

def norm_confint(x, ci=0.95):
    """
    to compute confidence interval based on normal distribution

    Parameters
    ----------
    x : ndarray
        input data.
    ci : float
        confidence level, ci = 0.95 for 95% confidence interval.

    Returns
    -------
    x_mean : ndarray
        mean value.
    lower : ndarray
        lower bound value.
    upper : ndarray
        upper bound value.

    """
    x_mean = np.mean(x,axis=0)
    x_sem = st.sem(x,axis=0)
    lower, upper = st.norm.interval(ci, loc=x_mean, scale=x_sem)        
    return x_mean, lower, upper
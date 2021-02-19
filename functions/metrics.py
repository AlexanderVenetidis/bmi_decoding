"""
Metrics for evaluating BMI performance
"""

import numpy as np

def compute_mse(ytrue,ypred):
    """
    Compute mean squared error (MSE) given the true and predicted values
    """
    mse_value = np.mean(np.square(ypred-ytrue),axis=0)
    return mse_value

def compute_rmse(ytrue,ypred):
    """
    Compute root mean squared error (RMSE) given the true and predicted values
    """
    mse_value = np.mean(np.square(ypred-ytrue),axis=0)
    rmse_value = np.sqrt(mse_value)
    return rmse_value

def compute_pearson(ytrue,ypred):
    """
    Compute Pearson's coefficient correlation given the true and predicted values
    """
    pearson_value = np.empty(ytrue.shape[1])
    pearson_value[:] = np.nan
    for i in range(ytrue.shape[1]): #Loop through outputs
        rho=np.corrcoef(ytrue[:,i],ypred[:,i],rowvar=False)[0,1] # choose the cross-covariance
        pearson_value[i] = rho
    return pearson_value
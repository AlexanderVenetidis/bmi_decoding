"""
List of preprocessing and utility functions
"""

import numpy as np

def flatten_list(lst):
    """
    to flatten a list
    """
    return [item for sublist in lst for item in sublist]

def flatten_array(arr):
    """
    to flatten an arrray
    """
    num_spike = 0
    for i in range(len(arr)):
        num_spike += arr[i].shape[-1]
        if arr[i].shape[0]>0:
            len_waveform = arr[i].shape[0]
    flat_arr = np.full((len_waveform,num_spike),np.nan)
    idx_spike = [0]
    for i in range(len(arr)):
        idx_spike.append(idx_spike[i]+arr[i].shape[-1])
        if arr[i].shape[0]>=0:
            flat_arr[:,idx_spike[i]:idx_spike[i+1]] = arr[i]
    return flat_arr

def split_index(X,num_fold):
    """
    to split input features into training, validation, and testing sets
    
    Parameters
    ----------
    X : ndarray
        input features to be split.
    num_fold : int
        number of folds.

    Returns
    -------
    all_train_idx : list
        train data indices.
    all_valid_idx : list
        validation data indices.
    all_test_idx : list
        testing data indices.

    """
    len_X = X.shape[0] # length of kinematic data
    idx_X = np.arange(len_X) # index of kinematic data    
    len_split_X = int(len_X/num_fold) # length of fold of data
    start_idx = 0
    all_split_idx = []
    for i in range(num_fold):
        split_idx = idx_X[start_idx:start_idx+len_split_X]
        all_split_idx.append(split_idx)
        start_idx += len_split_X
    all_sample_idx = np.asarray(all_split_idx)

    valid_idx = np.arange(num_fold) # 0,1,2,...,num_fold
    test_idx = np.roll(valid_idx,num_fold-1)
    
    all_train_idx = []
    all_valid_idx = []
    all_test_idx = []
    for i in range(num_fold):
        valid_temp_idx = all_sample_idx[valid_idx[i],:]
        test_temp_idx = all_sample_idx[test_idx[i],:]
        train_temp_idx = np.delete(all_sample_idx,[valid_idx[i],test_idx[i]],axis=0)
        train_temp_idx = train_temp_idx.flatten()
        all_train_idx.append(train_temp_idx)
        all_valid_idx.append(valid_temp_idx)
        all_test_idx.append(test_temp_idx)
    return all_train_idx,all_valid_idx,all_test_idx

        
def input_shaping(Xin,timestep,stride):
    """
    to shape input features by taking into account the previous samples
    Parameters
    ----------
    Xin : ndarray
        input features.
    timestep : int
        number of timesteps (current + previous timesteps).
    stride : int
        number of strides.

    Returns
    -------
    Xout : ndarray
        shaped input features.

    """
    num_sample = Xin.shape[0]
    num_feature = Xin.shape[1]
    num_iter = (num_sample - timestep)//stride + 1
    # initialise output data with nan values
    Xout = np.empty([num_iter,timestep,num_feature]) 
    Xout[:] = np.nan
    for i in range(num_iter): 
        start_idx = i*stride
        end_idx = start_idx + timestep
        Xout[i,:,:] = Xin[start_idx:end_idx,:] 
    return Xout
"""
Evaluating spike-based BMI decoding using Kalman filter
"""

# import packages
import h5py
import numpy as np
from functions.preprocess import input_shaping, split_index
from functions.decoders import kalman_decoder
from functions.metrics import compute_rmse, compute_pearson
import time as timer

print ("Starting simulation")
run_start = timer.time()

feature_list = ['sua_rate','mua_rate']
feature = feature_list[1]

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'          # file name
kinematic_folder = 'kinematic_data/'    # kinematic data folder
feature_folder = 'spike_data/features/' # spike features folder
result_folder = 'results/'              # results folder

wdw_time = 0.256 # window size in second
lag = -32 # lag between kinematic and feature data (minus indicate feature lagging behaind kinematic)
delta_time = 0.004 # sampling interval in second
wdw_samp = int(round(wdw_time/delta_time))
ol_samp = wdw_samp-1

# open spike features from hdf5 file
feature_file = feature_folder+file_name+'_spike_features_'+str(int(wdw_time*1e3))+'ms.h5'
print ("Loading input features from file: "+feature_file)
with h5py.File(feature_file,'r') as f:
    input_feature = f[feature].value

# open kinematic data from hdf5 file
kinematic_file = kinematic_folder+file_name+'_kinematic_data.h5'
print ("Loading kinematic data from file: "+kinematic_file)
with h5py.File(kinematic_file,'r') as f:
    cursor_pos = f['cursor_pos'].value #in mm
    cursor_vel = f['cursor_vel'].value # in mm/s
    cursor_acc = f['cursor_acc'].value # in mm/s^2

# set Kalman filter parameters
regular = 'l2' # regularisation type
alpha_reg = 0  # regularisation constant
params = {'regular': regular, 'alpha_reg':alpha_reg}

input_dim = input_feature.shape[1] # input dimension
output_dim = cursor_vel.shape[1] # output dimension
num_fold = 10 # number of folds

# initialise performance scores (RMSE and CC) with nan values
rmse_valid = np.full((num_fold,output_dim),np.nan)
rmse_test = np.copy(rmse_valid)
cc_valid = np.copy(rmse_valid)
cc_test = np.copy(rmse_valid)
time_train = np.full((num_fold),np.nan)
time_test = np.copy(time_train) 

print ("Formatting input feature data")
X_in = input_feature

print ("Formatting output (kinematic) data")
Y_out = np.concatenate((cursor_pos,cursor_vel,cursor_acc),axis=1)

print ("Splitting input dataset into training, validation, and testing subdataset")
all_train_idx,all_valid_idx,all_test_idx = split_index(Y_out,num_fold)

for i in range(num_fold):    
    train_idx = all_train_idx[i]
    valid_idx = all_valid_idx[i]
    test_idx = all_test_idx[i]
    
    # specify training dataset
    X_train = X_in[train_idx,:]            
    Y_train = Y_out[train_idx,:]
    
    # specify validation dataset
    X_valid = X_in[valid_idx,:]
    Y_valid = Y_out[valid_idx,:]
    
    # specify validation dataset
    X_test = X_in[test_idx,:]
    Y_test = Y_out[test_idx,:]
    
    # Standardise (z-score) input dataset
    X_train_mean = np.nanmean(X_train,axis=0)
    X_train_std = np.nanstd(X_train,axis=0) 
    X_train = (X_train - X_train_mean)/X_train_std 
    X_valid = (X_valid - X_train_mean)/X_train_std 
    X_test = (X_test - X_train_mean)/X_train_std 
    
    # Zero mean (centering) output dataset
    Y_train_mean = np.nanmean(Y_train,axis=0) 
    Y_train = Y_train - Y_train_mean 
    Y_valid = Y_valid - Y_train_mean
    Y_test = Y_test - Y_train_mean
           
    #Re-align data to take lag into account
    if lag < 0:
        X_train = X_train[:lag,:] # remove lag first from end (X lag behind Y)
        Y_train = Y_train[-lag:,:] # reomve lag first from beginning
        X_valid = X_valid[:lag,:]
        Y_valid = Y_valid[-lag:,:]
        X_test = X_test[:lag,:]
        Y_test = Y_test[-lag:,:]
    if lag > 0:
        X_train = X_train[lag:,:] # reomve lag first from beginning
        Y_train = Y_train[:-lag,:] # remove lag first from end (X lead in front of Y)
        X_valid = X_valid[lag:,:]
        Y_valid = Y_valid[:-lag,:]            
        X_test = X_test[lag:,:]
        Y_test = Y_test[:-lag,:]  
    
    print("Instantiating and training model...")    
    model = kalman_decoder() # instantiate model
    start = timer.time()
    model.fit(X_train,Y_train,**params) # train model
    end = timer.time()
    print("Model training took {:.2f} seconds".format(end - start))        
    time_train[i] = end - start
    print("Evaluating model...")
    Y_valid_predict = model.predict(X_valid,Y_valid)
    start = timer.time()
    Y_test_predict = model.predict(X_test,Y_test)
    end = timer.time()
    print("Model testing took {:.2f} seconds".format(end - start)) 
    time_test[i] = end - start
    
    # Compute performance metrics using velocity information    
    rmse_vld = compute_rmse(Y_valid[:,2:4],Y_valid_predict[:,2:4])
    rmse_tst = compute_rmse(Y_test[:,2:4],Y_test_predict[:,2:4])
    cc_vld = compute_pearson(Y_valid[:,2:4],Y_valid_predict[:,2:4])
    cc_tst = compute_pearson(Y_test[:,2:4],Y_test_predict[:,2:4])
    rmse_valid[i,:] = rmse_vld
    rmse_test[i,:] = rmse_tst
    cc_valid[i,:] = cc_vld
    cc_test[i,:] = cc_tst
        
    print("Fold-{} | Validation RMSE: {:.2f}".format(i,np.mean(rmse_vld)))
    print("Fold-{} | Validation CC: {:.2f}".format(i,np.mean(cc_vld)))
    print("Fold-{} | Testing RMSE: {:.2f}".format(i,np.mean(rmse_tst)))
    print("Fold-{} | Testing CC: {:.2f}".format(i,np.mean(cc_tst)))

run_end = timer.time()
mean_rmse_valid = np.nanmean(rmse_valid,axis=0)
mean_rmse_test = np.nanmean(rmse_test,axis=0)
mean_cc_valid = np.nanmean(cc_valid,axis=0)
mean_cc_test = np.nanmean(cc_test,axis=0)
mean_time =  np.nanmean(time_train,axis=0)
print("----------------------------------------------------------------------")
print("Mean validation RMSE: {:.2f}".format(np.mean(mean_rmse_valid)))
print("Mean validation CC: {:.2f}".format(np.mean(mean_cc_valid)))
print("Mean testing RMSE: {:.2f}".format(np.mean(mean_rmse_test)))
print("Mean testing CC: {:.2f}".format(np.mean(mean_cc_test)))
print("----------------------------------------------------------------------")

# storing evaluation results into hdf5 file
result_filename = result_folder+file_name+'_kf_'+feature+'_'+str(int(wdw_time*1e3))+'ms.h5'
print ("Storing results into file: "+result_filename)
with h5py.File(result_filename,'w') as f:
    f['Y_true'] = Y_test
    f['Y_predict'] = Y_test_predict
    f['rmse_valid'] = rmse_valid
    f['rmse_test'] = rmse_test
    f['cc_valid'] = cc_valid
    f['cc_test'] = cc_test
    f['time_train'] = time_train
    f['time_test'] = time_test

run_time = run_end - run_start
print ("Finished whole processes within %.2f seconds" % run_time)

"""
Evaluating spike-based BMI decoding using long short-term memory (LSTM) network
"""

# import packages
import h5py
import numpy as np
from functions.preprocess import input_shaping, split_index
from functions.decoders import lstm_decoder
from functions.metrics import compute_rmse, compute_pearson
import time as timer
from tensorflow import random

seed = 2020 # random seed for reproducibility

print ("Starting simulation")
run_start = timer.time()

feature_list = ['sua_rate','mua_rate']
feature = feature_list[1] # select which spike feature: SUA=0, MUA=1

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
#fields: ['cursor_acc', 'cursor_pos', 'cursor_time', 'cursor_vel', 'target_pos']
with h5py.File(kinematic_file,'r') as f:
    cursor_vel = f['cursor_vel'].value # in mm/s

print(input_feature.shape, cursor_vel.shape)

print(input_feature[:5])

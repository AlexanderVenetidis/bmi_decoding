"""
A module to store kinematic data
"""

# import required packages
import numpy as np
import h5py
import os

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'         # file name
raw_folder = 'raw_data/'               # raw data folder
kinematic_folder = 'kinematic_data/'  # kinematic data folder

# open raw data data from .mat file
mat_filename = raw_folder+file_name+'.mat'
print ("Loading kinematic data from file: {}".format(mat_filename))
with h5py.File(mat_filename,'r') as f:
    cursor_pos = f['cursor_pos'].value # 2 x N dimension, where N = number of samples
    cursor_pos = np.transpose(cursor_pos) # transpose to be N x 2 dimension
    target_pos = f['target_pos'].value # 2 x N dimension, where N = number of samples
    target_pos = np.transpose(target_pos) # transpose to be N x 2 dimension
    cursor_time = f['t'].value # 1 x N dimension, where N = number of samples
    cursor_time = np.squeeze(cursor_time) #times associated with cursor position, squeeze format data into 1D

dt_cursor = 0.004 # sampling period; difference between consecutive samples in cursor_time

# calculate cursor velocity and acceleration
temp_vel = np.diff(cursor_pos,axis=0) / dt_cursor # in mm/s
cursor_vel = np.concatenate((temp_vel,temp_vel[-1:,:]),axis=0) # padding with the last element
temp_acc = np.diff(cursor_vel,axis=0) / dt_cursor # in mm/s^2
cursor_acc = np.concatenate((temp_acc,temp_acc[-1:,:]),axis=0) # padding with the last element 

# store kinematic data into hdf5 format file
filename = kinematic_folder+file_name+'_kinematic_data.h5'
print ("Storing kinematic data into file: "+filename)
with h5py.File(filename,'w') as f:
    f['cursor_time'] = cursor_time
    f['cursor_pos'] = cursor_pos
    f['cursor_vel'] = cursor_vel
    f['cursor_acc'] = cursor_acc
    f['target_pos'] = target_pos
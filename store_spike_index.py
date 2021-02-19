"""
A module to store spike index for long-term tracking/decoding
"""

# import required packages
import numpy as np
import h5py
from functions.preprocess import flatten_list

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'          # file name
raw_folder = 'raw_data/'                # raw data folder
kinematic_folder = 'kinematic_data/'  # kinematic data folder
index_folder = 'spike_data/index/'      # spike index folder

# open kinematic data from hdf5 file
kinematic_filename = kinematic_folder+file_name+'_kinematic_data.h5'
print ("Loading kinematic data into file: "+kinematic_filename)
with h5py.File(kinematic_filename,'r') as f:
    cursor_time = f['cursor_time'].value

# open raw data from .mat file
mat_filename = raw_folder+file_name+'.mat'
print ("Loading spike data from file: "+mat_filename)
with h5py.File(mat_filename,'r') as f:
    spikes = f['spikes'].value # 5 x C dimension, where C = number of channels
    spikes = np.transpose(spikes) # transpose to be C x 5 dimension
    num_chan = spikes.shape[0]
    num_unit = spikes.shape[1]
    all_spikes = [] # list 96 channels of list 5 units containing array of num of spikes
    for i in range(num_chan):
        chan_spikes = []
        for j in range(num_unit):
            if (f[spikes[i,j]].ndim==2):
                temp_spikes = np.squeeze(f[spikes[i,j]].value,axis=0) # dimension: num of spikes, remove first dimension axis=0
            else:
                temp_spikes = np.empty(0)
            chan_spikes.append(temp_spikes)
        all_spikes.append(chan_spikes)

# Computing maximum number of spikes for the whole duration (max spike rate)
# Storing SUA index for long-term tracking
min_spikerate = 0.5
task_duration = cursor_time[-1]-cursor_time[0]
min_numspike = int(np.round(min_spikerate*task_duration))

all_num_spike = []
all_sua_idx = []
for i in range(num_chan):
    sua_idx = []
    for j in range(1,num_unit): # only sorted unit considered, unsorted (unit=0) excluded
        sua_train = all_spikes[i][j]
        if (sua_train.ndim > 0):
            sua_crop = np.where((sua_train>=cursor_time[0]) & (sua_train<=cursor_time[-1]))[0]
            sua_train = sua_train[sua_crop]
            if (len(sua_train) > min_numspike):
                sua_idx.append(j)
                all_num_spike.append(len(sua_train))
    all_sua_idx.append(sua_idx)
num_sua = len(all_num_spike)

# Storing MUA number for long-term tracking
mua_trains = []   
for i in range(num_chan):
    flat_mua = flatten_list(all_spikes[i])
    flat_mua.sort()
    mua_trains.append(np.asarray(flat_mua))    

mua_idx = []
for i in range(num_chan):
    mua_train = mua_trains[i]
    if (mua_train.ndim > 0):
        mua_crop = np.where((mua_train>=cursor_time[0]) & (mua_train<=cursor_time[-1]))[0]
        mua_train = mua_train[mua_crop]
        if (len(mua_train) > min_numspike):
            mua_idx.append(i)
        else:
            mua_idx.append(np.nan)
    else:
        mua_idx.append(np.nan)  

# determining which channels are bad (less than minimum spike rate threshold)
mua_idx_array = np.asarray(mua_idx)    
num_mua = np.sum(~np.isnan(mua_idx_array))    
mua_idx = np.squeeze(np.argwhere(~np.isnan(mua_idx_array)))
bad_channel = np.squeeze(np.argwhere(np.isnan(mua_idx_array)))

min_numspike = min(all_num_spike)/task_duration # min number of spike per second --> 25
max_numspike = max(all_num_spike)/task_duration # max number of spike per second --> 25
mean_numspike = sum(all_num_spike)/(len(all_num_spike)*task_duration) # mean number of spike per second --> 5        

# store spike index into hdf5 format file
filename = index_folder+file_name+'_spike_index.h5'
print ("Storing tracked SUA and MUA indices into file: "+filename)
with h5py.File(filename,'w') as f:
    dt = h5py.special_dtype(vlen=np.dtype('i1'))
    sua_index = f.create_dataset('sua_index', (num_chan,), dtype=dt)
    for i in range(num_chan):
        sua_index[i]=all_sua_idx[i]
    f['mua_index'] = mua_idx
    f['bad_channel'] = bad_channel
    f['min_spikerate'] = min_spikerate
    
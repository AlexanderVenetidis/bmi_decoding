"""
A module to store spike data (SUA and MUA)
"""

# import required packages
import numpy as np
import h5py
from functions.preprocess import flatten_list

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'          # file name
raw_folder = 'raw_data/'                # raw data folder
kinematic_folder = 'kinematic_data/'    # kinematic data folder
index_folder = 'spike_data/index/'      # spike index folder
spike_folder = 'spike_data/spikes/'      # spike data folder

# open kinematic data from hdf5 file
kinematic_filename = kinematic_folder+file_name+'_kinematic_data.h5'
print ("Loading kinematic data into file: "+kinematic_filename)
with h5py.File(kinematic_filename,'r') as f:
    cursor_time = f['cursor_time'].value

# open raw data from .mat file
mat_filename = raw_folder+file_name+'.mat'
print ("Loading kinematic data from file: "+mat_filename)
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

# Loading SUA indices for long-term decoding
index_filename = index_folder+file_name+'_spike_index.h5'
print ("Loading spike index data into file: "+index_filename)
with h5py.File(index_filename,'r') as f:
    sua_idx = f['sua_index'].value
    mua_idx = f['mua_index'].value
print(sua_idx.shape, mua_idx.shape)

sua_trains = []
num_sua_trains = []
for i in range(num_chan):
    if (len(sua_idx[i])>0):
        for j in range(len(sua_idx[i])):
            idx = sua_idx[i][j]
            sua_train = all_spikes[i][idx]
            sua_crop = np.where((sua_train>=cursor_time[0]) & (sua_train<=cursor_time[-1]))[0]
            sua_train = sua_train[sua_crop]
            sua_trains.append(sua_train)
            num_sua_trains.append(len(sua_train))
num_sua = len(sua_trains)

# Computing MUA or threshold crossings
num_mua = len(mua_idx)
mua_trains = []
num_mua_trains = []
for i in range(num_mua):
    idx = mua_idx[i]
    flat_mua = flatten_list(all_spikes[idx])
    flat_mua.sort()
    mua_train = np.asarray(flat_mua)
    mua_crop = np.where((mua_train>=cursor_time[0]) & (mua_train<=cursor_time[-1]))[0]
    mua_train = mua_train[mua_crop]
    mua_trains.append(mua_train)
    num_mua_trains.append(len(mua_train))

# store spike data into hdf5 format file
filename = spike_folder+file_name+'_spike_data.h5'
print ("Storing spike data into file: "+filename)
with h5py.File(filename,'w') as f:
    f['num_sua_trains'] = num_sua_trains
    f['num_mua_trains'] = num_mua_trains
    dt = h5py.special_dtype(vlen=np.dtype('f8'))
    sua_data = f.create_dataset('sua_trains', (num_sua,), dtype=dt)
    mua_data = f.create_dataset('mua_trains', (num_mua,), dtype=dt)
    for i in range(num_sua):
        sua_data[i] = sua_trains[i]
    for i in range(num_mua):
        mua_data[i] = mua_trains[i]

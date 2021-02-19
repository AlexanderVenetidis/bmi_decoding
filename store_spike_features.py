"""
A module to extract spike rate (count) from SUA or MUA trains using binning method
"""

# import packages
import numpy as np
from functions.extract_features import binning
import h5py
import time as timer

print ("Starting spike feature extraction...")
run_start = timer.time()

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'          # file name
kinematic_folder = 'kinematic_data/'    # kinematic data folder
spike_folder = 'spike_data/spikes/'     # spike data folder
feature_folder = 'spike_data/features/' # spike features folder

# open kinematic data from hdf5 file
kinematic_filename = kinematic_folder+file_name+'_kinematic_data.h5'
print ("Loading kinematic data into file: "+kinematic_filename)
with h5py.File(kinematic_filename,'r') as f:
    cursor_time = f['cursor_time'].value

# open spike data from hdf5 file
filename = spike_folder+file_name+'_spike_data.h5'
print ("Loading spike data from file: "+filename)
with h5py.File(filename,'r') as f:
    sua_train = f['sua_trains'].value #sorted spike times (single unit activity)
    mua_train = f['mua_trains'].value #unsorted spike times (threshold crossing/multi unit activity)

len_time = len(cursor_time)
num_sua = len(sua_train)
num_mua = len(mua_train)

wdw_time = 0.256 # window size in second
delta_time = 0.004 # sampling interval in second
wdw_samp = int(wdw_time/delta_time)
ol_samp = wdw_samp-1
sua_rate = np.full((len_time,num_sua),np.nan)
mua_rate = np.full((len_time,num_mua),np.nan)

for i in range(num_sua):
    print ("Extracting SUA/sorted spike features from unit no: %d" %i)
    sua = binning(sua_train[i],cursor_time,wdw_samp,ol_samp)
    sua_rate[:,i] = sua[:len_time]
for i in range(num_mua):
    print ("Extracting MUA/threshold crossing features from channel no: %d" %i)
    mua = binning(mua_train[i],cursor_time,wdw_samp,ol_samp)
    mua_rate[:,i] = mua[:len_time]

# store spike features into hdf5 format file
filename = feature_folder+file_name+'_spike_features_'+str(int(wdw_time*1e3))+'ms.h5'
print ("Storing spike features into file: "+filename)
with h5py.File(filename,'w') as f:
    f['sua_rate'] = sua_rate
    f['mua_rate'] = mua_rate

run_end = timer.time()
print ("Finished whole processes within %.2f seconds" % (run_end-run_start))

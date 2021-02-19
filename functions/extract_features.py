"""
To extract features (spike rate/count) from spike trains
"""
import numpy as np

def binning(spike_times,raw_times,wdw,ol):
    """
    Parameters
    ----------
    spike_times : ndarray
        array of spike times.
    raw_times : ndarray
        times at which spike rate being estimated.
    wdw : int
        window size in number of samples.
    ol : int
        number of overlap samples.

    Returns
    -------
    spike_rate : ndarray
        estimated spike rate.

    """
    num_samples = len(raw_times) # number of samples
    n_iter = num_samples // (wdw-ol) + 1 # number of iterations
    dt_times = raw_times[1] - raw_times[0] # sampling period
    # input data is padded to keep the length the same
    num_start_pad = wdw//2
    num_end_pad = wdw - num_start_pad
    start_pad = raw_times[0] - np.arange(num_start_pad,0,-1)*dt_times
    end_pad = raw_times[-1] + np.arange(1,num_end_pad + 1)*dt_times
    raw_times_pad = np.concatenate((start_pad,raw_times,end_pad))
    # initialise with nan values
    spike_rate = np.full((n_iter),np.nan)
    # count number of spikes in each bin for each neuron, and put in array
    for j in range(n_iter):
        idx_start = j*(wdw-ol)
        idx_end = idx_start + wdw
        # to check spike times within moving window size
        idxs = np.where((spike_times >= raw_times_pad[idx_start]) & (spike_times < raw_times_pad[idx_end-1]))[0] 
        temp_spike_times = spike_times[idxs]
        n_spikes = len(temp_spike_times)
        spike_rate[j] = n_spikes
    return spike_rate[:num_samples]
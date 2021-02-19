Directory and file structure for this project:
BMI-Decoding
├───functions
│   └───conf_interval.py		# module to compute confidence interval based on t or normal distribution
│   └───decoders.py				# list of decoding algorithm modules
│   └───extract_features.py		# module to extract features (spike rate/count) from spike trains
│   └───generator.py			# module to use customised data generator for training model in Keras
│   └───metrics.py				# module to compute performance metrics
│   └───preprocess.py			# list of preprocessing and utility functions
├───kinematic_data
├───plots
├───raw_data
├───results
└───spike_data
    ├───features
    ├───index
    └───spikes
├───README						# this file
└───store_kinematic.py			# to store kinematic data
├───store_spike_index.py		# to store spike index for long-term tracking/decoding
└───store_spike_data.py			# to store spike data (SUA and MUA)
├───store_spike_features.py		# to extract spike rate (count) from SUA or MUA trains using binning method
└───kf_spike_eval.py			# to evaluate spike-based BMI decoding using Kalman filter
├───wf_spike_eval.py			# to evaluate spike-based BMI decoding using Wiener filter
└───lstm_spike_eval.py			# to evaluate spike-based BMI decoding using long short-term memory (LSTM) network
├───plot_spike_eval.py			# to plot performance benchmark across decoding algorithms and signal inputs

Make sure to download raw neural data (.nwb) and kinematic data (.mat) from https://zenodo.org/record/583331 and store it in raw_data folder. Using above directory/file structure, run the following scripts:
1. store_kinematic.py
   Select the filename to be processed. In this directory, I only put one file as an example. You can add any file from https://zenodo.org/record/583331 by your self.
2. store_spike_index.py		
   This script is used when we are interested to track the activity from the same neuron over time or evaluate the long-term decoding performance from the same neuron. Here I used 0.5 Hz as spike rate threshold.
   Any channel with spike rate less than 0.5 Hz is categorised as bad channel and will be excluded for further processing.
3. store_spike_data.py			
   This script is used to store spike data (SUA and MUA).
4. store_spike_features.py		
   This script is used to extract spike rate (count) from SUA or MUA trains using binning method. Here I used overlapping window with size of 256 ms and generate spike features every 4 ms to match the timescale of the kinematic data.
   You can change different window size and overlap size.
5. <kf/wf/lstm>_spike_eval.py			
   This script is to evaluate spike-based BMI decoding using Kalman filter, Wiener filter, or long short-term memory (LSTM) network. You can play with the hyperparameters of each decoding algorithm to get better performance.
8. plot_spike_eval.py
   This script is to plot performance benchmark across decoding algorithms (KF, WF, and LSTM), signal inputs (SUA and MUA), and metrics (RMSE and CC).

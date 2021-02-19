"""
Plot performance benchmark across decoding algorithms and signal inputs
"""
import numpy as np
import h5py
from functions.conf_interval import t_confint
import matplotlib.pyplot as plt

feature_list = ['sua_rate','mua_rate']
decoder_list = ['kf','wf','lstm']

# specify filename to be processed (choose from the list available at https://zenodo.org/record/583331)
file_name = 'indy_20160915_01'          # file name
result_folder = 'results/'              # results folder
plot_folder = 'plots/'                  # plots folder
wdw_time = 0.256 # window size in second

# open performance score from file
len_feature = len(feature_list)
len_decoder = len(decoder_list)
all_rmse = []
all_cc = []
for i in range(len_feature):
    decoder_rmse = []
    decoder_cc = []
    for j in range(len_decoder):
        result_filename = result_folder+file_name+'_'+decoder_list[j]+'_'+feature_list[i]+'_'+str(int(wdw_time*1e3))+'ms.h5'
        with h5py.File(result_filename,'r') as f:
            rmse_test = f['rmse_test'].value
            cc_test = f['cc_test'].value # shape => num_fold=10,num_dim=2
        decoder_rmse.append(np.mean(rmse_test,axis=1))
        decoder_cc.append(np.mean(cc_test,axis=1))
    all_rmse.append(decoder_rmse)
    all_cc.append(decoder_cc)

# compute confidence interval
all_rmse_mean, all_rmse_lower, all_rmse_upper = [], [], []
all_cc_mean, all_cc_lower, all_cc_upper = [], [], []
for i in range(len(feature_list)):
    rmse_mean, rmse_lower, rmse_upper = [], [], []
    cc_mean, cc_lower, cc_upper = [], [], []
    for j in range(len(decoder_list)): 
        mean, lower, upper = t_confint(all_rmse[i][j],ci=0.95)
        rmse_mean.append(mean)
        rmse_lower.append(lower)
        rmse_upper.append(upper)
        mean, lower, upper = t_confint(all_cc[i][j],ci=0.95)
        cc_mean.append(mean)
        cc_lower.append(lower)
        cc_upper.append(upper)
    all_rmse_mean.append(rmse_mean)
    all_rmse_lower.append(rmse_lower)
    all_rmse_upper.append(rmse_upper)
    all_cc_mean.append(cc_mean)
    all_cc_lower.append(cc_lower)
    all_cc_upper.append(cc_upper)
    
metric_mean = [np.array(all_rmse_mean),np.array(all_cc_mean)]
metric_interval = [np.array(all_rmse_mean)-np.array(all_rmse_lower), 
                   np.array(all_cc_mean)-np.array(all_cc_lower)] # mean-lower = upper-mean
metric_names = ['RMSE','CC']
feature_names = ['SUA','MUA']
decoder_names = ['KF', 'WF', 'LSTM']
width = 0.3  # the width of the bars
x_loc = np.arange(len(decoder_names))  # the label locations
# plot performance comparison
fig, ax = plt.subplots(1,len(metric_names),figsize=(10,6))
for i in range(len(metric_names)):
    for j in range(len(feature_names)):
        ax[i].bar(x_loc+j*width, metric_mean[i][j,:], width, label=feature_names[j])
        ax[i].errorbar(x_loc+j*width, metric_mean[i][j,:], yerr=metric_interval[i][j,:], linestyle='None', color='k',capsize=3)
    ax[i].set_ylabel(metric_names[i],fontsize=14)
    ax[i].set_xticks(x_loc+(width/2))
    ax[i].set_xticklabels(decoder_names)
    ax[i].tick_params(axis='both', which='major', labelsize=12)
    ax[i].set_xlabel("Decoding algorithm",fontsize=14)
    ax[i].legend(loc='upper center',ncol=len_feature,fontsize=12)
#plt.show()
fig.tight_layout()
# store plot into .pdf file
plot_filename = plot_folder+'comparison.pdf'

print('saving..')
fig.savefig(plot_filename, bbox_inches='tight')
print('successful!')
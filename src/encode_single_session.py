import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utils.utils import set_seed
from utils.eval_utils import bits_per_spike
from scipy.ndimage import gaussian_filter1d
from models.encoders.models import train_model, train_model_main
from sklearn.linear_model import Ridge

temp = np.load('/burg/stats/users/sw3894/copy_from_server/yizi/data_new/db4df448_data_dict.npz', allow_pickle=True)
eid = "db4df448"

train_data = {eid: {"X": [], "y": [], 'setup': {"uuids": temp['train'].item()['cluster_uuids']}}}
T = 100; smooth_w = 2
for k in ['train', 'val', 'test']:
    def _one_hot(arr):
        uni = np.sort(np.unique(arr))
        ret = np.zeros((len(arr), T, len(uni)))
        for i, _uni in enumerate(uni):
            ret[:,:,i] = (arr == _uni)
        return ret
    # X = np.concatenate([_one_hot(temp[k].item()[v]) for v in ['block', 'choice', 'reward']], axis=2)
    # X = np.concatenate([X, temp[k].item()['dynamic_behavior']], axis=2)
    X = temp[k].item()['dynamic_behavior'].astype(np.float64)
    y = temp[k].item()['spikes_data']
    y = gaussian_filter1d(y, smooth_w, axis=1)  # (K, T, N)
    train_data[eid]['X'].append(X)
    train_data[eid]['y'].append(y)


train_data[eid]['X'] = [np.concatenate([train_data[eid]['X'][0], train_data[eid]['X'][1]], axis=0), train_data[eid]['X'][2]]
train_data[eid]['y'] = [np.concatenate([train_data[eid]['y'][0], train_data[eid]['y'][1]], axis=0), train_data[eid]['y'][2]]


def _std(arr):
    mean = np.mean(arr, axis=0) # (T, N)
    std = np.std(arr, axis=0) # (T, N)
    std = np.clip(std, 1e-8, None) # (T, N) 
    arr = (arr - mean) / std
    return arr, mean, std

_, mean_X, std_X = _std(train_data[eid]['X'][0])
_, mean_y, std_y = _std(train_data[eid]['y'][0])
for i in range(2):
    K = train_data[eid]['X'][i].shape[0]
    train_data[eid]['X'][i] = np.concatenate([(train_data[eid]['X'][i]-mean_X)/std_X, np.ones((K,T,1))], axis=2)
    train_data[eid]['y'][i] = (train_data[eid]['y'][i]-mean_y)/std_y

train_data[eid]['setup']['mean_y_TN'] = mean_y
train_data[eid]['setup']['std_y_TN'] = std_y
train_data[eid]['setup']['mean_X_Tv'] = mean_X
train_data[eid]['setup']['std_X_Tv'] = std_X



### RRR 
l2 = 100; n_comp = 4
model, mse_val = train_model_main(None,
                            train_data, l2, n_comp, None,
                            1., 5000, 1e-7, 1e-9, 100, None,
                            False, False, None, False)

_, _, pred_orig = model.predict_y_fr(train_data, eid, 1)
pred_orig = pred_orig.cpu().detach().numpy()


### FRR
l2 = 100.
X_train, X_test = train_data[eid]['X']
y_train, y_test = train_data[eid]['y']
reg_model = Ridge(l2, fit_intercept=False).fit(X_train[:,:,:-1].reshape((-1, X_train.shape[-1]-1)), 
                                        y_train.reshape((-1, y_train.shape[-1]))) 
K_test = X_test.shape[0]
pred_orig = reg_model.predict(X_test[:,:,:-1].reshape((K_test*T, X_test.shape[-1]-1)))
pred_orig = pred_orig.reshape((K_test, T, -1))
pred_orig = pred_orig * train_data[eid]['setup']['std_y_TN'] + train_data[eid]['setup']['mean_y_TN']



for thres in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
# for thres in [1e-3]:
    pred = np.clip(pred_orig, thres, None)
    bps = bits_per_spike(pred, temp['test'].item()['spikes_data'])
    print(thres, bps)


# ### viz
# X_all, y_all, y_all_pred = model.predict_y_fr(train_data, eid, 0)
# X_all = X_all.cpu().detach().numpy()
# y_all = y_all.cpu().detach().numpy()
# y_all_pred = y_all_pred.cpu().detach().numpy()

# from utils.plot import viz_single_cell, plot_single_trial_activity
# import matplotlib.pyplot as plt
# import os
# nis = np.where(mse_val['r2s_val'][eid]>0.3)[0].tolist()
# print(nis)
# for ni in nis:
#     print(ni)
#     X = X_all.copy()
#     X[:,:,:-2] = np.round(X_all[:,:,:-2], 0)
#     y = y_all[:,:,ni]/0.01
#     y_pred = y_all_pred[:,:,ni]/0.01
#     viz_single_cell(X, y, y_pred, "temp", 
#                     {"block": [0,1,2], "choice": [3,4], "reward":[5,6],}, ['block', 'choice','reward'], None, [],
#                     subtract_psth=None, aligned_tbins=[19], clusby='y_pred')
#     plt.savefig(os.path.join("temp", f"{ni}_{mse_val['r2s_val'][eid][ni]:.2f}_train.pdf")); plt.close('all')


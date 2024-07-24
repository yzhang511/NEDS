import os
import pickle
import argparse
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
import wandb
import torch
import matplotlib.pyplot as plt
from datasets import (
    load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
)
from utils.dataset_utils import (
    get_user_datasets, load_ibl_dataset, split_both_dataset
)
from loader.make_loader import make_loader
from utils.utils import set_seed, huggingface2numpy, _one_hot, _std 
from utils.config_utils import config_from_kwargs, update_config
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from utils.eval_utils import bits_per_spike, viz_single_cell
from models.rrr_encoder import train_model, train_model_main


# -----------
# USER INPUTS
# -----------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="db4df448-e449-4a6f-a0e7-288711e7a75a")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--encode_static_behavior", action="store_true")
ap.add_argument("--l2_penalty", type=int, default=100)
ap.add_argument("--rank", type=int, default=4)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--save_plot", action="store_true")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
args = ap.parse_args()


# ------
# CONFIG
# ------
eid = args.eid
base_path = args.base_path
avail_beh = args.behavior
avail_mod = args.modality
kwargs = {"model": f"include:src/configs/baseline.yaml"}
config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/trainer.yaml", config)
modal_filter = {
    "input": ['behavior'], 
    "output": ['ap']
}
log_dir = os.path.join(
    base_path, "results", f"ses-{eid}", "set-train",
    f"{args.num_sessions}_sessions",
    f"inModal-{'-'.join(modal_filter['input'])}",
    f"outModal-{'-'.join(modal_filter['output'])}",
    args.model,
)
os.makedirs(log_dir, exist_ok=True)

save_path = os.path.join(
    base_path,
    "results",
    f"ses-{eid}",
    "set-eval",
    f"{args.num_sessions}_sessions",
    f"inModal-{'-'.join(modal_filter['input'])}",
    f"outModal-{'-'.join(modal_filter['output'])}",
    args.model,
)
set_seed(config.seed)

if args.wandb:
    wandb.init(
        project="multi_modal",
        config=args,
        name="ses-{}_set-eval_{}-sessions_inModal-{}_outModal{}-model-{}".format(
            eid[:5], 
            args.num_sessions,
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            args.model,
    )
)

# ---------
# LOAD DATA
# ---------
_, _, _, meta_data = load_ibl_dataset(
    config.dirs.dataset_cache_dir, 
    config.dirs.huggingface_org,
    eid=eid,
    num_sessions=1,
    split_method="predefined",
    test_session_eid=[],
    batch_size=config.training.train_batch_size,
    seed=config.seed
)

dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
train_dataset, val_dataset, test_dataset = dataset["train"], dataset["val"], dataset["test"]

n_behaviors, n_neurons = len(avail_beh), len(train_dataset['cluster_regions'][0])
meta_data['num_neurons'] = [n_neurons]
print(meta_data)

train_dataloader = make_loader(
    train_dataset, 
    target=avail_beh,
    load_meta=config.data.load_meta,
    batch_size=config.training.train_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=n_neurons,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    shuffle=True
)

val_dataloader = make_loader(
    val_dataset, 
    target=avail_beh,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=n_neurons,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    shuffle=False
)

test_dataloader = make_loader(
    test_dataset, 
    target=avail_beh,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=n_neurons,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    shuffle=False
)

# ------------
# PREPARE DATA
# ------------
data_dict = huggingface2numpy(
    train_dataloader, val_dataloader, test_dataloader, test_dataset
)

train_data = {
    eid: {"X": [], "y": [], "setup": {"uuids": data_dict['train']['cluster_uuids']}}
}

smooth_w = 2; T = config.data.max_time_length
for k in ["train", "test"]:
    if args.encode_static_behavior:
        X = np.concatenate(
            [_one_hot(data_dict[k][v]) for v in ["block", "choice", "reward"]], axis=2
        )
        X = np.concatenate([X, data_dict[k]["dynamic_behavior"]], axis=2)
    X = data_dict[k]["dynamic_behavior"].astype(np.float64)
    y = data_dict[k]["spikes_data"]
    y = gaussian_filter1d(y, smooth_w, axis=1)  # (K, T, N)
    train_data[eid]["X"].append(X)
    train_data[eid]["y"].append(y)

_, mean_X, std_X = _std(train_data[eid]["X"][0])
_, mean_y, std_y = _std(train_data[eid]["y"][0])
for i in range(2):
    K = train_data[eid]["X"][i].shape[0]
    train_data[eid]["X"][i] = np.concatenate(
        [(train_data[eid]["X"][i]-mean_X)/std_X, np.ones((K,T,1))], axis=2
    )
    train_data[eid]["y"][i] = (train_data[eid]["y"][i]-mean_y)/std_y

train_data[eid]["setup"]["mean_y_TN"] = mean_y
train_data[eid]["setup"]["std_y_TN"] = std_y
train_data[eid]["setup"]["mean_X_Tv"] = mean_X
train_data[eid]["setup"]["std_X_Tv"] = std_X


# --------
# TRAINING
# --------
l2 = args.l2_penalty; n_comp = args.rank

if args.model == "rrr":
    model, mse_val = train_model_main(
        train_data, l2, n_comp, "tmp", save=True
    )
    _, _, pred_orig = model.predict_y_fr(train_data, eid, 1)
    pred_orig = pred_orig.cpu().detach().numpy()

elif args.model == "linear":
    X_train, X_test = train_data[eid]["X"]
    y_train, y_test = train_data[eid]["y"]
    reg_model = Ridge(l2, fit_intercept=False).fit(
        X_train[:,:,:-1].reshape((-1, X_train.shape[-1]-1)), 
        y_train.reshape((-1, y_train.shape[-1]))
    ) 
    K_test = X_test.shape[0]
    pred_orig = reg_model.predict(
        X_test[:,:,:-1].reshape((K_test*T, X_test.shape[-1]-1))
    )
    pred_orig = pred_orig.reshape((K_test, T, -1))
    pred_orig = pred_orig * train_data[eid]["setup"]["std_y_TN"] + \
                train_data[eid]["setup"]["mean_y_TN"]
else:
     raise NotImplementedError


# ----
# EVAL
# ----
threshold = 1e-3
pred_held_out = np.clip(pred_orig, threshold, None)
gt_held_out = data_dict["test"]["spikes_data"]

bps_result_list = []
for n_i in tqdm(range(n_neurons), desc='co-bps'):     
    bps = bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])
    if np.isinf(bps):
        bps = np.nan
    bps_result_list.append(bps)
# bps_result_list.append(bits_per_spike(pred_held_out, gt_held_out))

# Bits per spike calculation
save_path = f"{save_path}/modal_spike"
os.makedirs(save_path, exist_ok=True)
bps_all = np.array(bps_result_list)
bps_mean, bps_std = np.nanmean(bps_all), np.nanstd(bps_all)
np.save(os.path.join(save_path, f"bps.npy"), bps_all)
results = {"mean_bps": bps_mean}

# Single-neuron visualization
if args.save_plot:
    X_all, y_all, y_all_pred = model.predict_y_fr(train_data, eid, 1)
    X_all = X_all.cpu().detach().numpy()
    y_all = y_all.cpu().detach().numpy()
    y_all_pred = y_all_pred.cpu().detach().numpy()   
    # nis = np.where(mse_val["r2s_val"][eid]>0.3)[0].tolist()
    nis = list(range(n_neurons))
    r2_result_list = []
    for ni in nis:
        X = X_all.copy()
        X[:,:,:-2] = np.round(X_all[:,:,:-2], 0)
        y = y_all[:,:,ni] #/0.01
        y_pred = y_all_pred[:,:,ni] #/0.01
        _r2_psth, _r2_trial = viz_single_cell(
            X, y, y_pred, "temp", 
            {"block": [0,1,2], "choice": [3,4], "reward":[5,6],}, 
            ["block", "choice","reward"], None, [],
            subtract_psth=None, aligned_tbins=[19], clusby="y_pred"
        )
        plt.savefig(os.path.join(save_path, f"{ni}_{mse_val['r2s_val'][eid][ni]:.2f}.png")); 
        plt.close('all')
        r2_result_list.append([_r2_psth, _r2_trial])

    r2_all = np.array(r2_result_list)
    r2_psth_mean = np.nanmean(r2_result_list.T[0]) 
    r2_trial_mean = np.nanstd(r2_result_list.T[1])
    np.save(os.path.join(save_path, f"r2.npy"), r2_all)
    results.update({
        "mean_r2_psth": r2_psth_mean,
        "mean_r2_trial": r2_trial_mean,
    })
    
print(results)
if args.wandb:
    wandb.log(results)



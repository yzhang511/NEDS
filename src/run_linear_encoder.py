import os
import wandb
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from utils.dataset_utils import load_ibl_dataset
from loader.make_loader import make_loader
from utils.utils import (
    set_seed, 
    huggingface2numpy, nlb2numpy,
    _one_hot, _std
)
from utils.config_utils import config_from_kwargs, update_config
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from utils.eval_utils import bits_per_spike, viz_single_cell
from models.rrr_encoder import train_model, train_model_main

logging.basicConfig(level=logging.INFO) 

kwargs = {"model": "include:src/configs/baseline.yaml"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/trainer_encoder.yaml", config)
set_seed(config.seed)

# ------ 
# SET UP
# ------ 
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--encode_static_behavior", action="store_true")
ap.add_argument("--rank", type=int, default=3)
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--save_plot", action="store_true")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--use_nlb", action="store_true")
ap.add_argument("--nlb_bin_size", type=int, default=5)
args = ap.parse_args()

eid = args.eid
base_path = args.base_path
avail_beh = args.behavior if not args.use_nlb else ["finger_vel", "cursor_pos", "finger_pos"]
avail_mod = args.modality
num_sessions = args.num_sessions
model_class = args.model
n_comp = args.rank
smooth_w = 2
trial_len, threshold = 2 if not args.use_nlb else 0.6, 1e-3

modal_filter = {"input": ["behavior"], "output": ["ap"]}

assert num_sessions == 1, \
    "Only support single-session. Refer to train_baseline.py for multi-session."

# --------
# SET PATH
# --------
log_name = "ses-{}_set-eval_{}-sessions_inModal-{}_outModal{}-model-{}".format(
    eid[:5] if not args.use_nlb else "nlb-rtt", 
    num_sessions,
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    model_class,
)
log_dir = os.path.join(base_path, "results", log_name)
os.makedirs(log_dir, exist_ok=True)
logging.info(f"Save model to {log_dir}")
save_path = log_dir.replace("train", "eval")

if args.wandb:
    wandb.init(
        project="baseline",
        config=args,
        name=log_name
    )

# ---------
# LOAD DATA
# ---------
if not args.use_nlb:
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
        config.dirs.dataset_cache_dir, 
        config.dirs.huggingface_org,
        eid=eid,
        num_sessions=1,
        split_method="predefined",
        test_session_eid=[],
        batch_size=config.training.train_batch_size,
        seed=config.seed,
    )
else:
    from utils.nlb_data_utils import load_nlb_dataset
    _trial_length, bin_size = 600, args.nlb_bin_size
    train_dataset, val_dataset, test_dataset, meta_data, behavior_means = load_nlb_dataset(
        "/projects/beez/yzhang39/nlb/000129/sub-Indy", bin_size, standardize=False
    )
    config["data"]["max_time_length"] = int(_trial_length / bin_size)

T = config.data.max_time_length
n_neurons = meta_data["eid_list"][eid]
meta_data["num_neurons"] = [n_neurons]
logging.info(meta_data)

local_data_dir = "ibl_mm" if args.num_sessions == 1 else f"ibl_mm_{args.num_sessions}"

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
    shuffle=True,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="train",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    use_nlb=args.use_nlb
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
    shuffle=False,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="val",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    use_nlb=args.use_nlb
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
    shuffle=False,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="test",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    use_nlb=args.use_nlb
)

# ---------------
# PREPROCESS DATA
# ---------------
PREPROCESS_FN = nlb2numpy if args.use_nlb else huggingface2numpy
data_dict = PREPROCESS_FN(
    train_dataloader, val_dataloader, test_dataloader, test_dataset
)
train_data = {
    eid: {"X": [], "y": [], "setup": {"uuids": data_dict["train"]["cluster_uuids"]} if not args.use_nlb else {}}
}
for k in ["train", "test"]:
    if args.encode_static_behavior:
        X = np.concatenate(
            [_one_hot(data_dict[k][v], T) for v in ["block", "choice"]], axis=2
        )
        X = np.concatenate([X, data_dict[k]["dynamic_behavior"]], axis=2)
    else:
        X = data_dict[k]["dynamic_behavior"].astype(np.float64)
        
    y = data_dict[k]["spikes_data"]
    y = gaussian_filter1d(y, smooth_w, axis=1)
    train_data[eid]["X"].append(X)
    train_data[eid]["y"].append(y)

"""WARNING: LEGACY CODE. DEBUGGING REQUIRED."""
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


# -----
# TRAIN
# -----

if model_class == "rrr":

    l2s = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
    n_comp_list = [1, 2, 3, 4, 5]
    best_mse_val = np.inf
    best_l2 = None
    best_n_comp = None
    for l2 in l2s:
        for n_comp in n_comp_list:
            model, mse_val = train_model_main(
                train_data, l2, n_comp, "tmp", save=True
            )
            if mse_val["mse_val_mean"] < best_mse_val:
                best_mse_val = mse_val["mse_val_mean"]
                best_model = model
                best_l2 = l2
                best_n_comp = n_comp
    print(f"Best L2: {best_l2}, Best n_comp: {best_n_comp}")
    _, _, pred_orig = best_model.predict_y_fr(train_data, eid, 1)
    pred_orig = pred_orig.cpu().detach().numpy()

elif model_class == "linear":

    X_train, X_test = train_data[eid]["X"]
    y_train, y_test = train_data[eid]["y"]

    reg_model = RidgeCV(
        alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4], fit_intercept=False
    )
    reg_model.fit(
        # X_train[..., :-1].reshape((-1, X_train.shape[-1]-1)), 
        # y_train.reshape((-1, y_train.shape[-1]))
        X_train[..., :-1].reshape((-1, T * (X_train.shape[-1]-1))), 
        y_train.reshape((-1, T * y_train.shape[-1]))
    ) 
    print(f"Selected alpha: {reg_model.alpha_}")
    K_test = X_test.shape[0]
    pred_orig = reg_model.predict(
        # X_test[..., :-1].reshape((K_test*T, X_test.shape[-1]-1))
        X_test[..., :-1].reshape((K_test, -1))
    )
    pred_orig = pred_orig.reshape((K_test, T, -1))
    pred_orig = pred_orig * train_data[eid]["setup"]["std_y_TN"] + \
                train_data[eid]["setup"]["mean_y_TN"]
else:
     raise NotImplementedError


# ----
# EVAL
# ----
pred_held_out = np.clip(pred_orig, threshold, None)
gt_held_out = data_dict["test"]["spikes_data"]
print("gt_held_out.shape", gt_held_out.shape)
mean_fr = gt_held_out.sum(1).mean(0) / trial_len
keep_idxs = np.arange(len(mean_fr)).flatten()

bps = bits_per_spike(pred_held_out, gt_held_out)
bps = np.nan if np.isinf(bps) else bps
bps_result_list = [bps]

# bps_result_list = []
# for n_i in tqdm(keep_idxs, desc="co-bps"):     
#     bps = bits_per_spike(pred_held_out[..., [n_i]], gt_held_out[..., [n_i]])
#     if np.isinf(bps):
#         bps = np.nan
#     bps_result_list.append(bps)

save_path = f"{save_path}/modal_spike"
os.makedirs(save_path, exist_ok=True)
bps_all = np.array(bps_result_list)
bps_mean, bps_std = np.nanmean(bps_all), np.nanstd(bps_all)
np.save(os.path.join(save_path, f"bps.npy"), bps_all)
results = {"mean_bps": bps_mean}

# Single-neuron visualization
if args.save_plot:
    for k in ["train", "test"]:
        X = np.concatenate(
            [_one_hot(data_dict[k][v], T) for v in ["block", "choice", "reward"]], axis=2
        ) 
    X_all, y_all, y_all_pred = model.predict_y_fr(train_data, eid, 1)
    X_all = X_all.cpu().detach().numpy()
    y_all = y_all.cpu().detach().numpy()
    y_all_pred = y_all_pred.cpu().detach().numpy()   
    nis = list(range(n_neurons))
    r2_result_list = []
    for ni in nis:
        X = X_all.copy()
        X[..., :-2] = np.round(X_all[..., :-2], 0)
        y = y_all[..., ni] 
        y_pred = y_all_pred[..., ni] 
        _r2_psth, _r2_trial = viz_single_cell(
            X, y, y_pred, "temp", 
            {"block": [0, 1, 2], "choice": [3, 4], "reward":[5, 6],}, 
            ["block", "choice","reward"], None, [],
            subtract_psth=None, aligned_tbins=[19], clusby="y_pred"
        )
        plt.savefig(os.path.join(save_path, f"{ni}_{mse_val['r2s_val'][eid][ni]:.2f}.png")); 
        plt.close("all")
        r2_result_list.append([_r2_psth, _r2_trial])

    r2_all = np.array(r2_result_list)
    r2_psth_mean = np.nanmean(r2_result_list.T[0]) 
    r2_trial_mean = np.nanstd(r2_result_list.T[1])
    np.save(os.path.join(save_path, "r2.npy"), r2_all)
    results.update({
        "mean_r2_psth": r2_psth_mean,
        "mean_r2_trial": r2_trial_mean,
    })
    
logging.info(results)
if args.wandb:
    wandb.log(results)



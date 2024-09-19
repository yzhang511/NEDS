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
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from utils.eval_utils import bits_per_spike, viz_single_cell
from sklearn.metrics import accuracy_score


# -----------
# USER INPUTS
# -----------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="db4df448-e449-4a6f-a0e7-288711e7a75a")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--model", type=str, default="linear", choices=["rrr", "linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--decode_static_behavior", action="store_true")
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--rank", type=int, default=3)
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
config = update_config(f"src/configs/trainer_decoder.yaml", config)
modal_filter = {
    "input": ['ap'], 
    "output": ['behavior']
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
        project="baseline_decoder",
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

n_behaviors, n_neurons = len(avail_beh), meta_data['eid_list'][eid]
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

T = config.data.max_time_length
for k in ["train", "test"]:
    if args.decode_static_behavior:
        X = np.concatenate(
            [_one_hot(data_dict[k][v], T) for v in ["block", "choice", "reward"]], axis=2
        )[:,0]
    else:
        X = data_dict[k]["dynamic_behavior"].astype(np.float64)[:,:,:2]
    y = data_dict[k]["spikes_data"]
    train_data[eid]["X"].append(X)
    train_data[eid]["y"].append(y)


# --------
# TRAINING
# --------
l2 = 100
n_comp = args.rank
if args.model == "rrr":
    raise NotImplementedError

elif args.model == "linear":
    X_train, X_test = train_data[eid]["y"]
    y_train, y_test = train_data[eid]["X"]
    pred = []
    for i in range(y_test.shape[-1]):
        if args.decode_static_behavior:
            clf_model = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]).fit(
                X_train.reshape((X_train.shape[0], -1)), 
                y_train[:,i]
            )
            K_test = X_test.shape[0]
            pred_orig = clf_model.predict(
                X_test.reshape((X_test.shape[0], -1))
            )
            pred.append(pred_orig)
        else:
            reg_model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]).fit(
                X_train.reshape((-1, X_train.shape[-1])), 
                y_train[:,:,i].flatten()
            ) 
            K_test = X_test.shape[0]
            pred_orig = reg_model.predict(
                X_test.reshape((-1, X_test.shape[-1]))
            )
            pred.append(pred_orig.reshape(K_test, T, -1))
else:
     raise NotImplementedError


# ----
# EVAL
# ----

results = {}

if not args.decode_static_behavior:
    
    for k in ["train", "test"]:
        X = np.concatenate(
            [_one_hot(data_dict[k][v], T) for v in ["choice", "reward", "block"]], axis=2
        )
    
    var_name2idx = {'block': [2], 'choice': [0], 'reward': [1]}
    var_value2label = {'block': {(0.2,): "p(left)=0.2", (0.5,): "p(left)=0.5", (0.8,): "p(left)=0.8",},
                       'choice': {(-1.0,): "right", (1.0,): "left"},
                       'reward': {(0.,): "no reward", (1.,): "reward", }}
    var_tasklist = ['block', 'choice', 'reward']
    var_behlist = []
    
    avail_beh = ['wheel-speed', 'whisker-motion-energy']
    for i in range(y_test.shape[-1]):
        r2_result_list = []
        y = y_test[:,:,[i]]
        y_pred = pred[i] 
        _r2_psth, _r2_trial = viz_single_cell(
            X, y, y_pred, 
            var_name2idx, var_tasklist, var_value2label, var_behlist, 
            subtract_psth='task', 
            aligned_tbins=[],
            neuron_idx=avail_beh[i],
            neuron_region='',
            method=args.model, 
            save_path=save_path,
            save_plot=False,
        )
        plt.close('all')
        results.update({
            f"{avail_beh[i]}_r2_psth": _r2_psth,
            f"{avail_beh[i]}_r2_trial": _r2_trial,
        })
else:
    choice_acc = accuracy_score(y_test[:,[0]], pred[0])
    reward_acc = accuracy_score(y_test[:,[1]], pred[1])
    block_acc = accuracy_score(y_test[:,[2]], pred[2])
    
    results.update({
        "choice_acc": choice_acc,
        "reward_acc": reward_acc,
        "block_acc": block_acc,
    })
    
print(results)
if args.wandb:
    wandb.log(results)



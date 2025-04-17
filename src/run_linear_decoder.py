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
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from utils.eval_utils import bits_per_spike, viz_single_cell
from sklearn.metrics import accuracy_score, balanced_accuracy_score

logging.basicConfig(level=logging.INFO) 

kwargs = {"model": f"include:src/configs/baseline.yaml"}
config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/trainer_decoder.yaml", config)
set_seed(config.seed)

# ------ 
# SET UP
# ------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--model", type=str, default="linear", choices=["linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--decode_static_behavior", action="store_true")
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

modal_filter = {"input": ["ap"], "output": ["behavior"]}

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
        seed=config.seed
    )
else:
    from utils.nlb_data_utils import load_nlb_dataset
    trial_length, bin_size = 600, args.nlb_bin_size # ms
    train_dataset, val_dataset, test_dataset, meta_data, behavior_means = load_nlb_dataset(
        "/projects/beez/yzhang39/nlb/000129/sub-Indy", bin_size
    )
    config["data"]["max_time_length"] = int(trial_length / bin_size)

    np.save(f"/u/yzhang39/multi_modal_foundation_model/train_dataset_{bin_size}.npy", train_dataset)
    np.save(f"/u/yzhang39/multi_modal_foundation_model/val_dataset_{bin_size}.npy", val_dataset)
    np.save(f"/u/yzhang39/multi_modal_foundation_model/test_dataset_{bin_size}.npy", test_dataset)
    np.save(f"/u/yzhang39/multi_modal_foundation_model/behavior_means_{bin_size}.npy", behavior_means)


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
    if args.decode_static_behavior:
        X = np.concatenate(
            [_one_hot(data_dict[k][v], T) for v in ["block", "choice", "reward"]], axis=2
        )[:,0]
    else:
        X = data_dict[k]["dynamic_behavior"].astype(np.float64)[..., :2]
    y = data_dict[k]["spikes_data"]
    smooth_w = 1 if args.use_nlb else 2
    y = gaussian_filter1d(y, smooth_w, axis=1)
    train_data[eid]["X"].append(X)
    train_data[eid]["y"].append(y)

# -----
# TRAIN
# -----

if args.model == "linear":
    X_train, X_test = train_data[eid]["y"]
    y_train, y_test = train_data[eid]["X"]
    pred = []
    for i in range(y_test.shape[-1]):
        if args.decode_static_behavior:
            clf_model = LogisticRegressionCV(
                Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
            ).fit(
                X_train.reshape((X_train.shape[0], -1)), 
                y_train[:, i]
            )
            K_test = X_test.shape[0]
            pred_orig = clf_model.predict(
                X_test.reshape((X_test.shape[0], -1))
            )
            pred.append(pred_orig)
        else:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import GridSearchCV
            # reg_model = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 0, 9)})
            reg_model = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 4, 9)})
            reg_model.fit(
                # X_train.reshape((-1, X_train.shape[-1])), 
                X_train.reshape((-1, T * X_train.shape[-1])), 
                # y_train[..., i].flatten()
                y_train[..., i]
            ) 
            K_test = X_test.shape[0]
            pred_orig = reg_model.predict(
                # X_test.reshape((-1, X_test.shape[-1]))
                X_test.reshape((-1, T * X_test.shape[-1]))
            )
            pred.append(pred_orig.reshape(K_test, T, -1))
else:
     raise NotImplementedError


# ----
# EVAL
# ----
results = {}
if not args.use_nlb:
    if not args.decode_static_behavior:
        for k in ["train", "test"]:
            X = np.concatenate(
                [_one_hot(data_dict[k][v], T) for v in ["choice", "reward", "block"]], axis=2
            )
        var_name2idx = {"block": [2], "choice": [0], "reward": [1]}
        var_value2label = {
            "block": {(0.2,): "p(left)=0.2", (0.5,): "p(left)=0.5", (0.8,): "p(left)=0.8",},
            "choice": {(-1.0,): "right", (1.0,): "left"},
            "reward": {(0.,): "no reward", (1.,): "reward", }
        }
        var_tasklist = ["block", "choice", "reward"]
        var_behlist = []
        
        for i in range(y_test.shape[-1]):
            r2_result_list = []
            y = y_test[..., [i]]
            y_pred = pred[i] 
            _r2_psth, _r2_trial = viz_single_cell(
                X, y, y_pred, 
                var_name2idx, var_tasklist, var_value2label, var_behlist, 
                subtract_psth="task", 
                aligned_tbins=[],
                neuron_idx=avail_beh[i],
                neuron_region="",
                method=args.model, 
                save_path=save_path,
                save_plot=False,
            )
            plt.close("all")
            results.update({
                f"{avail_beh[i]}_r2_psth": _r2_psth,
                f"{avail_beh[i]}_r2_trial": _r2_trial,
            })
    else:
        choice_acc = accuracy_score(y_test[:,[0]], pred[0])
        reward_acc = accuracy_score(y_test[:,[1]], pred[1])
        block_acc = accuracy_score(y_test[:,[2]], pred[2])
        choice_balance_acc = balanced_accuracy_score(y_test[:,[0]], pred[0])
        reward_balance_acc = balanced_accuracy_score(y_test[:,[1]], pred[1])
        block_balance_acc = balanced_accuracy_score(y_test[:,[2]], pred[2])
        
        results.update({
            "choice_acc": choice_acc,
            "reward_acc": reward_acc,
            "block_acc": block_acc,
            "choice_balance_acc": choice_balance_acc,
            "reward_balance_acc": reward_balance_acc,
            "block_balance_acc": block_balance_acc,
        })
else:
    _, _, test_1ms, meta_data, _ = load_nlb_dataset(
        "/projects/beez/yzhang39/nlb/000129/sub-Indy", 1
    )
    y_test = test_1ms["finger_vel"]

    np.save(f"/u/yzhang39/multi_modal_foundation_model/test_1ms.npy", test_1ms)
    
    r2_result_list = []
    for i in range(y_test.shape[-1]):
        y, y_pred = y_test[..., [i]], pred[i] 
        print(y_pred.max(), y_pred.min())
        y_pred = y_pred + behavior_means["test"]["finger_vel"][...,i]
        y_pred = np.repeat(y_pred[...,None], bin_size, axis=1).squeeze(-1)
        from sklearn.metrics import r2_score
        _r2_trial = r2_score(y.flatten(), y_pred.flatten())
        r2_result_list.append(_r2_trial)
        from matplotlib import pyplot as plt
        n_bin_to_plot = 10000
        plt.figure(figsize=(20, 3))
        plt.plot(y.flatten()[:n_bin_to_plot], label="GT")
        plt.plot(y_pred.flatten()[:n_bin_to_plot], label="Pred")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"pred_finger_vel_dim_{i}.png"))
        np.save(os.path.join(save_path, f"pred_finger_vel_dim_{i}.npy"), {"gt": y, "pred": y_pred})

    results.update({
        f"{avail_beh}_r2_trial": np.nanmean(r2_result_list),
    })
    
logging.info(results)
if args.wandb:
    wandb.log(results)

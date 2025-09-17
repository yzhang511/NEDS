import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from one.api import ONE

from datasets import DatasetDict, DatasetInfo

from utils.ibl_data_utils import (
    prepare_data,
    select_brain_regions,
    list_brain_regions,
    bin_spiking_data,
    bin_behaviors,
    align_data
)
from utils.dataset_utils import create_dataset, upload_dataset
from utils.preprocess_lfp import prepare_lfp, featurize_lfp

logging.basicConfig(level=logging.INFO) 

np.random.seed(42)

# ------ 
# SET UP
# ------
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--huggingface_org", type=str, default="ibl-repro-ephys")
ap.add_argument("--use_lfp", action="store_true")
ap.add_argument("--n_sessions", type=int, default=1)
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--eid", type=str)
args = ap.parse_args()

if args.n_sessions == 1:
    if args.eid is None:
        raise ValueError("Session EID is required.")
    else:
        eids = [args.eid]
else:
    PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f"{PROJ_DIR}/data/eids.txt") as file:
        eids = [line.rstrip() for line in file][:args.n_sessions]

params = {
    "interval_len": 2, 
    "binsize": 0.02, 
    "single_region": False,
    "align_time": 'stimOn_times', 
    "time_window": (-.5, 1.5), 
    "fr_thresh": 0.2
}

beh_names = [
    "choice", "reward", "block",
    "wheel-speed", "whisker-motion-energy", 
]

DYNAMIC_VARS = list(filter(lambda x: x not in ["choice", "reward", "block"], beh_names))

# ---------------
# PREPROCESS DATA
# ---------------
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international", 
    silent=True,
    cache_dir=args.base_path
)

final_eids = []
for eid_idx, eid in enumerate(eids):

    # if os.path.exists(f"{args.base_path}/{eid}_aligned"):
    #     logging.info(f"The dataset {eid}_aligned already exists.")
    #     continue
    
    logging.info(f"EID {eid}")

    neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data(
        one, eid, params, n_workers=args.n_workers
    )

    if neural_dict is None:
        logging.info(f"Skip EID {eid} Due to Missing Spike Data!")
        continue

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    bin_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, 
        neural_dict, 
        trials_df=trials_dict["trials_df"], 
        n_workers=args.n_workers, 
        **params
    )

    logging.info(f"Binned Spike Data: {bin_spikes.shape}")

    # Keep responsive neurons
    mean_fr = bin_spikes.sum(1).mean(0) / params["interval_len"]
    keep_unit_idxs = np.argwhere(mean_fr > 1/params["fr_thresh"]).flatten()
    bin_spikes = bin_spikes[..., keep_unit_idxs]
    logging.info(
        f"# Responsive Units: {bin_spikes.shape[-1]} / {len(mean_fr)}"
    )

    meta_dict["cluster_regions"] = [meta_dict["cluster_regions"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_channels"] = [meta_dict["cluster_channels"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_depths"] = [meta_dict["cluster_depths"][idx] for idx in keep_unit_idxs]
    meta_dict["good_clusters"] = [meta_dict["good_clusters"][idx] for idx in keep_unit_idxs]
    meta_dict["uuids"] = [meta_dict["uuids"][idx] for idx in keep_unit_idxs]
    # meta_dict["cluster_qc"] = {
    #     k: np.asarray(v)[keep_unit_idxs].tolist() for k, v in meta_dict["cluster_qc"].items()
    # }

    bin_beh, beh_mask = bin_behaviors(
        one, 
        eid, 
        DYNAMIC_VARS, 
        trials_df=trials_dict["trials_df"], 
        allow_nans=True, 
        n_workers=args.n_workers, 
        **params,
    )

    if args.use_lfp:
        lfp_prec = prepare_lfp(one, eid, dead_channel_threshold=0., **params)
        all_psd = featurize_lfp(
            lfp_prec, bin_size=int(params["interval_len"]/params["binsize"])
        )
        bin_lfp = []
        for lfp_band in all_psd.values():
            bin_lfp.append(lfp_band)
        bin_lfp = np.concatenate(bin_lfp, -1)
        logging.info(f"Binned LFP Data: {bin_lfp.shape}")
    else:
        bin_lfp = None

    try:
        align_bin_spikes, align_bin_beh, align_bin_lfp, target_mask, bad_trial_idxs = align_data(
            bin_spikes, 
            bin_beh, 
            bin_lfp, 
            list(bin_beh.keys()), 
            trials_dict["trials_mask"], 
        )
    except ValueError as e:
        logging.info(f"Skip EID {eid} due to error: {e}")
        continue

    if "whisker-motion-energy" not in align_bin_beh:
        logging.info(f"Skip EID {eid} due to missing whisker data.")
        continue

    # Data partition (train: 0.7 val: 0.1 test: 0.2)
    num_trials = len(align_bin_spikes)
    trial_idxs = np.random.choice(np.arange(num_trials), num_trials, replace=False)

    trial_mask = np.array(target_mask).astype(bool).tolist()
    trials_dict['trials_df'] = trials_dict['trials_df'][trial_mask]
    intervals = np.vstack([
        trials_dict['trials_df'][params["align_time"]] + params["time_window"][0],
        trials_dict['trials_df'][params["align_time"]] + params["time_window"][1]
    ]).T
    intervals = intervals[trial_idxs]

    train_idxs = trial_idxs[:int(0.7*num_trials)]
    val_idxs = trial_idxs[int(0.7*num_trials):int(0.8*num_trials)]
    test_idxs = trial_idxs[int(0.8*num_trials):]

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in align_bin_beh.keys():
        train_beh.update({beh: align_bin_beh[beh][train_idxs]})
        val_beh.update({beh: align_bin_beh[beh][val_idxs]})
        test_beh.update({beh: align_bin_beh[beh][test_idxs]})

    train_dataset = create_dataset(
        align_bin_spikes[train_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=train_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[train_idxs]
    )

    val_dataset = create_dataset(
        align_bin_spikes[val_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=val_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[val_idxs]
    )

    test_dataset = create_dataset(
        align_bin_spikes[test_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=test_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[test_idxs]
    )

    dataset = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )
    logging.info(dataset)

    # upload_dataset(dataset, org=args.huggingface_org, eid=f"{eid}_aligned")
    dataset.save_to_disk(f"{args.base_path}/{eid}_aligned")

    logging.info(f"Downloaded EID: {eid}")
    logging.info(f"Progress: {eid_idx+1} / {len(eids)} sessions downloaded")

    final_eids.append(eid)

logging.info(f"Successfully downloaded EIDs: ")

for eid in final_eids:
    print(eid)

import os
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.special import gammaln
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    r2_score, 
    balanced_accuracy_score, 
    accuracy_score
)

import torch
from accelerate import Accelerator
from datasets import load_from_disk, concatenate_datasets

from loader.make_loader import make_loader
from utils.dataset_utils import load_ibl_dataset, get_binned_spikes_from_sparse

from utils.utils import (
    set_seed, 
    move_batch_to_device, 
    plot_gt_pred, 
    metrics_list, 
    plot_avg_rate_and_spike, 
    plot_rate_and_spike, 
    plot_neurons_r2,
)
from utils.config_utils import config_from_kwargs, update_config

from multi_modal.mm import MultiModal
from multi_modal.encoder_embeddings import EncoderEmbedding

NAME2MODEL = {"MultiModal": MultiModal}

logger = logging.getLogger(__name__)

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel", "whisker"]

neural_acronyms = {
    "ap": "spike",
}
static_acronyms = {
    "choice": "choice",
    "block": "block",
}
dynamic_acronyms = {
    "wheel-speed": "wheel",
    "whisker-motion-energy": "whisker",
}

OUTPUT_DIM = {
    "choice": 2, 
    "block": 3, 
    "wheel": 1, 
    "whisker": 1, 
}

# --------------------------------------------------------------------------------------------------
# Model/Dataset Loading and Configuration
# --------------------------------------------------------------------------------------------------
def load_model_data_local(**kwargs):
    seed = kwargs["seed"]
    eid = kwargs["eid"]
    model_config = kwargs["model_config"]
    trainer_config = kwargs["trainer_config"]
    model_path = kwargs["model_path"]
    dataset_path = kwargs["dataset_path"]
    mask_name = kwargs["mask_name"]
    mask_mode = mask_name.split("_")[1]
    modal_filter = kwargs["modal_filter"]
    model_mode = kwargs["model_mode"]
    neural_mods = kwargs["neural_mods"]
    static_mods = kwargs["static_mods"]
    dynamic_mods = kwargs["dynamic_mods"]
    search = kwargs.get("search", False)

    num_sessions = kwargs["num_sessions"] if "num_sessions" in kwargs else 1

    avail_mod = neural_mods + static_mods + dynamic_mods
    avail_beh = static_mods + dynamic_mods

    set_seed(seed)

    if isinstance(model_config, str):
        config = config_from_kwargs({"model": f"include:{model_config}"})
        config = update_config(model_config, config)
        config = update_config(trainer_config, config)
    elif isinstance(model_config, dict):
        config = model_config

    _, _, dataset, meta_data = load_ibl_dataset(
        config.dirs.dataset_cache_dir, 
        config.dirs.huggingface_org,
        num_sessions=num_sessions,
        eid = eid if num_sessions == 1 else None,
        use_re=True,
        split_method="predefined",
        test_session_eid=[],
        batch_size=config.training.train_batch_size,
        seed=config.seed
    )

    logger.info(meta_data)

    if search:
        substring = "results/"
        first_index = model_path.find(substring)
        second_index = model_path.find(substring, first_index + len(substring))
        if second_index != -1: 
            model_path = model_path[:second_index] + model_path[second_index + len(substring):]
        else:
            model_path = model_path

        import pickle
        with open(f"{model_path.replace('model_best_avg.pt', '')}/params.pkl", "rb") as file:
            params = pickle.load(file)

        config["model"]["encoder"]["transformer"]["hidden_size"] = params["hidden_size"]
        config["model"]["encoder"]["transformer"]["inter_size"] = params["inter_size"]
        config["model"]["encoder"]["transformer"]["n_layers"] = params["n_layers"]

    encoder_embeddings = {}
    hidden_size = config.model.encoder.transformer.hidden_size
    for mod in modal_filter["input"]:
        encoder_embeddings[mod] = EncoderEmbedding(
            hidden_size = hidden_size,
            n_channel = hidden_size,
            output_channel = hidden_size,
            stitching = True,
            eid_list = meta_data["eid_list"],
            mod = mod,
            config = config.model.encoder,
            max_F = config.data.max_time_length,
        )

    NAME2MODEL = {"MultiModal": MultiModal}
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(
        encoder_embeddings,
        avail_mod = neural_mods + static_mods + dynamic_mods,
        avail_beh = static_mods + dynamic_mods,
        model_mode = model_mode,
        config = config.model, 
        **config.method.model_kwargs, 
        **meta_data
    )

    try:
        state_dict = torch.load(model_path)["model"]
    except:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))["model"]

    model.load_state_dict(state_dict) 
    
    # Change model to eval mode
    model.masker.ratio = 0
    model.masker.mask_regions = []
    model.masker.target_regions = []

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    _, _, dataset, meta_data = load_ibl_dataset(
        config.dirs.dataset_cache_dir, 
        config.dirs.huggingface_org,
        num_sessions=1,
        eid = eid,
        use_re=True,
        split_method="predefined",
        test_session_eid=[],
        batch_size=config.training.train_batch_size,
        seed=config.seed
    )

    n_neurons = max(list(meta_data["eid_list"].values()))

    logger.info(f"Load test data with {n_neurons} neurons.")
    
    dataloader = make_loader(
        dataset, 
        target=dynamic_mods,
        batch_size=10000,
        pad_to_right=True, pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=n_neurons,
        dataset_name=config.data.dataset_name,
        load_meta=config.data.load_meta,
        shuffle=False,
        seed=config.seed,
        data_dir=f"{kwargs['data_path']}/ibl_mm",
        mode="test",
        eids=list(meta_data["eids"]),
        stitching=False,
    )

    return model, accelerator, dataset, dataloader
    

# --------------------------------------------------------------------------------------------------
# Evaluation
# 1. Spiking activity reconstruction
# 2. Behavior reconstruction
# 3. Co-smooth/forward-pred/inter-region/intra-region
# --------------------------------------------------------------------------------------------------
def co_smoothing_eval(
    model,
    accelerator,
    test_dataloader,
    test_dataset,
    save_plot=False,
    is_multimodal=False,
    trial_len=2,
    fr_threshold=1,
    **kwargs
):
    for batch in test_dataloader:
        break

    mode = kwargs["mode"]
    T = kwargs["n_time_steps"]
    method_name = kwargs["method_name"]
    is_aligned = kwargs["is_aligned"]
    target_regions = kwargs["target_regions"]

    N = sum(batch["space_attn_mask"][0] != 0)
        
    uuids_list = np.array(test_dataset["cluster_uuids"][0])[:N]
    region_list = np.array(test_dataset["cluster_regions"])[0][:N]

    substring = "results/"
    first_index = kwargs['save_path'].find(substring)
    second_index = kwargs['save_path'].find(substring, first_index + len(substring))
    if second_index != -1: 
        kwargs['save_path'] = kwargs['save_path'][:second_index] + kwargs['save_path'][second_index + len(substring):]
    else:
        kwargs['save_path'] = kwargs['save_path']

    os.makedirs(kwargs["save_path"], exist_ok=True)

    if is_aligned:
        b_list = []
    
        choice = np.array(test_dataset["choice"])
        choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, T))
        b_list.append(choice)
    
        reward = np.array(test_dataset["reward"])
        reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, T))
        b_list.append(reward)
    
        block = np.array(test_dataset["block"])
        block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, T))
        b_list.append(block)
    
        behavior_set = np.stack(b_list, axis=-1)
    
        var_name2idx = {"block": [2], "choice": [0], "reward": [1], "wheel": [3]}
        var_value2label = {
            "block": {(0.2,): "p(left)=0.2", (0.5,): "p(left)=0.5", (0.8,): "p(left)=0.8",},
            "choice": {(-1.0,): "right", (1.0,): "left"},
            "reward": {(0.,): "no reward", (1.,): "reward", }
        }
        var_tasklist = ["block", "choice", "reward"]
        var_behlist = []
    
    if mode == "eval_spike":

        held_out_list = [kwargs["held_out_list"]]
        assert held_out_list[0] is not None, \
            "Forward prediction requires specific target time points to predict."
        target_regions = neuron_regions = None

        bps_result_list = [float("nan")] * N
        r2_result_list = [np.array([np.nan, np.nan])] * N
        
        for hd_idx in held_out_list:

            hd = np.array([hd_idx])

            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    
                    mask_result = heldout_mask(
                        batch["spikes_data"].clone(),
                        mode=mode,
                        heldout_idxs=hd,
                        target_regions=target_regions,
                        neuron_regions=region_list
                    )  

                    all_ones = torch.ones_like(
                        batch["spikes_data"]).to(accelerator.device, torch.int64)
                    all_zeros = all_ones * 0.

                    mod_dict = {}
                    for mod in model.mod_to_indx.keys():
                        mod_dict[mod] = {}
                        mod_idx = model.mod_to_indx[mod]
                        mod_dict[mod]["inputs_modality"] = torch.tensor(mod_idx).to(accelerator.device)
                        mod_dict[mod]["targets_modality"] = torch.tensor(mod_idx).to(accelerator.device)
                        mod_dict[mod]["inputs_attn_mask"] = batch["time_attn_mask"]
                        mod_dict[mod]["inputs_timestamp"] = batch["spikes_timestamps"]
                        mod_dict[mod]["targets_timestamp"] = batch["spikes_timestamps"]
                        # Each batch contains samples from different sessions
                        mod_dict[mod]["eid"] = batch["eid"]
                        mod_dict[mod]["num_neuron"] = N
                        mod_dict[mod]["training_mode"] = "encoding"

                        if mod == "spike":
                            mod_dict[mod]["inputs"] = batch["spikes_data"].clone()
                            mod_dict[mod]["targets"] = batch["spikes_data"].clone()
                        elif mod in model.avail_beh:
                            mod_dict[mod]["inputs"] = batch[mod].clone()
                            mod_dict[mod]["targets"] = batch[mod].clone()
                        else:
                           raise Exception(f"Modality {mod} not implemented.")

                        mod_dict[mod]["eval_mask"] = all_ones \
                            if not is_multimodal and mod == "spike" else all_zeros

                if is_multimodal:
                    for mod in model.mod_to_indx.keys():
                        mod_dict[mod]["eval_mask"] = all_ones if mod == "spike" else all_zeros

                # Mask selected modalities for encoding
                if "enc_task_var" in kwargs:
                    if kwargs["enc_task_var"] not in ["all", "random"]:
                        for mod in model.mod_to_indx:
                            mod_dict[mod]["inputs_token_mask"] = all_zeros if mod == kwargs["enc_task_var"] else all_ones
                    
                outputs = model(mod_dict)
                    
            gt = outputs.mod_targets["spike"][...,:N]
            preds = torch.exp(outputs.mod_preds["spike"][...,:N])
            gt = gt.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            target_n_i, target_t_i = np.arange(N), held_out_list[0]

            gt_held_out = gt[:,target_t_i][...,target_n_i]
            pred_held_out = preds[:,target_t_i][...,target_n_i]

            for n_i in tqdm(range(len(target_n_i)), desc="co-bps"): 
                bps = bits_per_spike(pred_held_out[...,[n_i]], gt_held_out[...,[n_i]])
                if np.isinf(bps):
                    bps = np.nan
                bps_result_list[target_n_i[n_i]] = bps

            ys, y_preds = gt[:,target_t_i], preds[:,target_t_i]

            np.save(
                f"{kwargs['save_path']}/spike_data.npy", {"gt": ys, "pred": y_preds}
            )
        
            for i in tqdm(range(target_n_i.shape[0]), desc="R2"):
                mean_fr = ys[..., target_n_i[i]].sum(1).mean(0) / trial_len
                if mean_fr >= 1/fr_threshold: 
                    if is_aligned:
                        X = behavior_set[:, target_t_i, :]  
                        _r2_psth, _r2_trial = viz_single_cell(
                            X, ys[...,target_n_i[i]], y_preds[...,target_n_i[i]],
                            var_name2idx, var_tasklist, var_value2label, var_behlist,
                            subtract_psth=kwargs["subtract"],
                            aligned_tbins=[],
                            neuron_idx=uuids_list[target_n_i[i]][:4],
                            neuron_region=region_list[target_n_i[i]],
                            method=method_name, save_path=kwargs["save_path"],
                            save_plot=save_plot
                            )
                        r2_result_list[target_n_i[i]] = np.array([_r2_psth, _r2_trial])
                    else:
                        raise ValueError("Unaligned data not supported.")
                
    elif mode == "eval_behavior":

        N = len(DYNAMIC_VARS)
        held_out_list = [kwargs["held_out_list"]]
        assert held_out_list[0] is not None, \
            "Forward prediction requires specific target time points to predict."
        target_regions = neuron_regions = None

        bps_result_list = [float('nan')] * N
        r2_result_list = [np.array([np.nan, np.nan])] * N
        
        for hd_idx in held_out_list:

            hd = np.array([hd_idx])

            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    mask_result_dict = {}
                    for mod in STATIC_VARS + DYNAMIC_VARS:
                        mask_result = heldout_mask(
                            batch[mod].clone(),
                            mode=mod,
                            heldout_idxs=hd
                        )
                        mask_result_dict[mod] = mask_result

                    all_ones = torch.ones_like(
                        batch["spikes_data"]).to(accelerator.device, torch.int64
                    )
                    all_zeros = all_ones * 0.

                    mod_dict = {}
                    for mod in model.mod_to_indx.keys():
                        mod_dict[mod] = {}
                        mod_idx = model.mod_to_indx[mod]
                        mod_dict[mod]["inputs_modality"] = torch.tensor(mod_idx).to(accelerator.device)
                        mod_dict[mod]["targets_modality"] = torch.tensor(mod_idx).to(accelerator.device)
                        mod_dict[mod]["inputs_attn_mask"] = batch["time_attn_mask"]
                        mod_dict[mod]["inputs_timestamp"] = batch["spikes_timestamps"]
                        mod_dict[mod]["targets_timestamp"] = batch["spikes_timestamps"]
                        # Each batch contains samples from different sessions
                        mod_dict[mod]["eid"] = batch["eid"]
                        mod_dict[mod]["num_neuron"] = N
                        mod_dict[mod]['training_mode'] = "decoding"

                        if mod == "spike":
                            mod_dict[mod]["inputs"] = batch["spikes_data"].clone()
                            mod_dict[mod]["targets"] = batch["spikes_data"].clone()
                        elif mod in model.avail_beh:
                            mod_dict[mod]["inputs"] = batch[mod].clone()
                            mod_dict[mod]["targets"] = batch[mod].clone()
                        else:
                           raise Exception(f"Modality {mod} not implemented.")

                        mod_dict[mod]["eval_mask"] = all_ones \
                            if not is_multimodal and mod in STATIC_VARS + DYNAMIC_VARS else all_zeros

                if is_multimodal:
                    for mod in model.mod_to_indx.keys():
                        mod_dict[mod]["eval_mask"] = \
                        all_ones if mod in model.avail_beh else all_zeros
                    
                outputs = model(mod_dict)
                                
            gt, preds = [], []
            gt_static, preds_static = {}, {}
            acc_dict, balanced_acc_dict = {}, {}
            for mod in outputs.mod_targets.keys():
                if mod in DYNAMIC_VARS:
                    gt.append(outputs.mod_targets[mod].detach().cpu().numpy())
                    preds.append(outputs.mod_preds[mod].detach().cpu().numpy())
                elif mod in STATIC_VARS:
                    gt_static[mod] = outputs.static_targets[mod].detach().cpu().numpy()
                    preds_static[mod] = outputs.static_preds[mod].detach().cpu().numpy()
              
            gt = np.stack(gt, axis=-1).squeeze(2)
            preds = np.stack(preds, axis=-1).squeeze(2)

            for mod in STATIC_VARS:
                acc_dict[mod] = accuracy_score(gt_static[mod], preds_static[mod])
                balanced_acc_dict[mod] = balanced_accuracy_score(gt_static[mod], preds_static[mod])

            target_n_i, target_t_i = np.arange(N), held_out_list[0]

            gt_held_out = gt[:,target_t_i][...,target_n_i]
            pred_held_out = preds[:,target_t_i][...,target_n_i]

            for n_i in tqdm(range(len(target_n_i)), desc="co-bps"): 
                bps_result_list[target_n_i[n_i]] = np.nan

            ys, y_preds = gt[:,target_t_i], preds[:,target_t_i]

            behav_results = {}
            for i in tqdm(range(target_n_i.shape[0]), desc="R2"):
                beh_name = DYNAMIC_VARS[i]
                if is_aligned:
                    X = behavior_set[:,target_t_i,:]  
                    _r2_psth, _r2_trial = viz_single_cell(
                        X, ys[...,target_n_i[i]], y_preds[...,target_n_i[i]],
                        var_name2idx, var_tasklist, var_value2label, var_behlist,
                        subtract_psth=kwargs["subtract"],
                        aligned_tbins=[],
                        neuron_idx=uuids_list[target_n_i[i]][:4],
                        neuron_region=region_list[target_n_i[i]],
                        method=method_name, save_path=kwargs["save_path"],
                        save_plot=save_plot
                    )
                    r2_result_list[target_n_i[i]] = np.array([_r2_psth, _r2_trial])
                    behav_results[f"{beh_name}_r2_psth"] = _r2_psth
                    behav_results[f"{beh_name}_r2_trial"] = _r2_trial

                    y = ys[...,target_n_i[i]].squeeze()
                    y_pred = y_preds[...,target_n_i[i]].squeeze()
                    np.save(
                        f"{kwargs['save_path']}/{beh_name}_data.npy", {"gt": y, "pred": y_pred}
                    )
                else:
                    raise ValueError("Unaligned data not supported.")
                    
            np.save(os.path.join(kwargs["save_path"], "r2.npy"), behav_results)
            np.save(
                os.path.join(kwargs["save_path"], "acc.npy"), 
                {"acc": acc_dict, "balanced_acc": balanced_acc_dict}
            )
            return {
                **behav_results,
                **acc_dict,
                **balanced_acc_dict
            }     
    else:
        raise NotImplementedError("Evaluation mode not implemented.")

    os.makedirs(kwargs["save_path"], exist_ok=True)
    bps_all = np.array(bps_result_list)
    bps_mean = np.nanmean(bps_all)
    r2_all = np.array(r2_result_list)
    np.save(os.path.join(kwargs["save_path"], "bps.npy"), bps_all)
    np.save(os.path.join(kwargs["save_path"], "r2.npy"), r2_all)

    # Mask selected modalities for encoding
    if "enc_task_var" in kwargs:
        if isinstance(kwargs["enc_task_var"], list):
            mode = f"eval_spike_{'_'.join(kwargs['enc_task_var'])}"
        else:
            mode = f"eval_spike_{kwargs['enc_task_var']}"

    return {
        f"{mode}_mean_bps": bps_mean,
        f"{mode}_mean_r2_psth": np.nanmean(r2_all[:, 0]),
        f"{mode}_mean_r2_trial": np.nanmean(r2_all[:, 1]),
    }


# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------
def heldout_mask(
    spike_data,                     # (K, T, N)
    mode='manual',                  
    heldout_idxs=np.array([]),      # list for manual mode
    n_active=1,                     # n_neurons for most-active mode
    target_regions=None,            # list for region mode
    neuron_regions=None,            # list for region mode
):
    mask = torch.ones(spike_data.shape).to(torch.int64).to(spike_data.device)

    if mode == 'manual':
        hd = heldout_idxs
        mask[:, :, hd] = 0

    elif mode == 'most':
        _act = spike_data.detach().cpu().numpy()
        _act = np.mean(_act, axis=(0, 1))
        act_idx = np.argsort(_act)
        hd = np.array(act_idx[-n_active:])
        mask[:, :, hd] = 0

    elif mode == 'inter_region':
        hd = []
        for region in target_regions:
            region_idxs = np.argwhere(neuron_regions == region).flatten()
            mask[:, :, region_idxs] = 0 
            target_idxs = region_idxs[heldout_idxs]
            hd.append(target_idxs)
        hd = np.stack(hd).flatten()

    elif mode == 'intra_region':
        mask *= 0
        hd = []
        for region in target_regions:
            region_idxs = np.argwhere(neuron_regions == region).flatten()
            mask[:, :, region_idxs] = 1 
            if len(heldout_idxs) == 0:
                target_idxs = region_idxs
            else:
                target_idxs = region_idxs[heldout_idxs]
                mask[:, :, target_idxs] = 0
            hd.append(target_idxs)
        hd = np.stack(hd).flatten()
            
    elif mode == 'forward_pred' or mode == 'eval_spike':
        hd = heldout_idxs
        mask[:, hd, :] = 0
    
    elif mode == 'eval_behavior':
        hd = heldout_idxs; n_heldout = len(np.unique(heldout_idxs))
        if mask.shape[1] != n_heldout:
            mask = mask.expand(-1, n_heldout, 1)
        mask[:, hd] = 0
    elif mode in DYNAMIC_VARS:
        hd = heldout_idxs
        mask[:, hd] = 0
    elif mode in STATIC_VARS:
        mask[:] = 0
        hd = 0

    else:
        raise NotImplementedError('mode not implemented')

    spike_data_masked = spike_data * mask

    return {"spikes": spike_data_masked, "heldout_idxs": hd, "eval_mask": 1-mask}

# --------------------------------------------------------------------------------------------------
# copied from NLB repo
# standard evaluation metrics
# --------------------------------------------------------------------------------------------------
def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


# --------------------------------------------------------------------------------------------------
# single neuron plot functions
# --------------------------------------------------------------------------------------------------
def create_behave_list(batch, T=100):
    
    b_list = []
        
    choice = batch['choice'].detach().cpu().numpy()
    choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, T))
    b_list.append(choice)
    
    reward = batch['reward'].detach().cpu().numpy()
    reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, T))
    b_list.append(reward)
    
    block = batch['block'].detach().cpu().numpy()
    block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, T))
    b_list.append(block)
    
    behavior_set = np.stack(b_list, axis=-1)
    
    var_name2idx = {'block': [2], 'choice': [0], 'reward': [1], 'wheel': [3]}
    var_value2label = {'block': {(0.2,): "p(left)=0.2", (0.5,): "p(left)=0.5", (0.8,): "p(left)=0.8", },
                           'choice': {(-1.0,): "right", (1.0,): "left"},
                           'reward': {(0.,): "no reward", (1.,): "reward", }}
    var_tasklist = ['block', 'choice', 'reward']
    var_behlist = []
    
    return behavior_set, var_name2idx, var_tasklist, var_value2label, var_behlist


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: for each task variable in var_tasklists, compute PSTH
:var_name2idx: for each task variable in var_tasklists, the corresponding index of X
:var_value2label:
:aligned_tbins: reference time steps to annotate. 
"""
def plot_psth(X, y, y_pred, var_tasklist, var_name2idx, var_value2label,
              aligned_tbins=[],
              axes=None, legend=False, neuron_idx='', neuron_region='', save_plot=False):
    
    if save_plot:
        if axes is None:
            nrows = 1;
            ncols = len(var_tasklist)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

        for ci, var in enumerate(var_tasklist):
            ax = axes[ci]
            psth_xy = compute_all_psth(X, y, var_name2idx[var])
            psth_pred_xy = compute_all_psth(X, y_pred, var_name2idx[var])
            
            for _i, _x in enumerate(psth_xy.keys()):
                
                psth = psth_xy[_x]
                psth_pred = psth_pred_xy[_x]
                ax.plot(psth,
                        color=plt.get_cmap('tab10')(_i),
                        linewidth=3, alpha=0.3, label=f"{var}: {tuple(_x)[0]:.2f}")
                ax.plot(psth_pred,
                        color=plt.get_cmap('tab10')(_i),
                        linestyle='--')
                ax.set_xlabel("Time bin")
                if ci == 0:
                    ax.set_ylabel("Neural activity")
                else:
                    ax.sharey(axes[0])
            _add_baseline(ax, aligned_tbins=aligned_tbins)
            if legend:
                ax.legend()
                ax.set_title(f"{var}")

    # compute PSTH for task_contingency
    idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
    psth_xy = compute_all_psth(X, y, idxs_psth)
    psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy, clip=False)
    r2_single_trial = compute_R2_main(y.reshape(-1, 1), y_pred.reshape(-1, 1), clip=False)[0]
    
    if save_plot:
        axes[0].set_ylabel(
            f'Neuron: #{neuron_idx[:4]} \n PSTH R2: {r2_psth:.2f} \n Avg_SingleTrial R2: {r2_single_trial:.2f}')

        for ax in axes:
            # ax.axis('off')
            ax.spines[['right', 'top']].set_visible(False)
            # ax.set_frame_on(False)
            # ax.tick_params(bottom=False, left=False)
        plt.tight_layout()

    return r2_psth, r2_single_trial


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: variables used for computing the task-condition-averaged psth if subtract_psth=='task'
:var_name2idx:
:var_tasklist: variables to be plotted in the single-trial behavior
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:aligned_tbins: reference time steps to annotate. 
:nclus, n_neighbors: hyperparameters for spectral_clustering
:cmap, vmax_perc, vmin_perc: parameters used when plotting the activity and behavior
"""
def plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth="task",
                               aligned_tbins=[],
                               n_clus=8, n_neighbors=5, n_pc=32, clusby='y_pred',
                               cmap='bwr', vmax_perc=90, vmin_perc=10,
                               axes=None):
    if axes is None:
        ncols = 1;
        nrows = 2 + len(var_behlist) + 1 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        psth_xy = compute_all_psth(X, y, idxs_psth)
        psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
        y_psth = np.asarray(
            [psth_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y_predpsth = np.asarray(
            [psth_pred_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_psth = np.mean(y, 0)
        y_predpsth = np.mean(y_pred, 0)
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"
    y_residual = (y_pred - y)  # (K, T), residuals of prediction
    idxs_behavior = np.concatenate(([var_name2idx[var] for var in var_behlist])) if len(var_behlist) > 0 else []
    X_behs = X[:, :, idxs_behavior]

    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0)
    if clusby == 'y_pred':
        clustering = clustering.fit(y_pred)
    elif clusby == 'y':
        clustering = clustering.fit(y)
    else:
        assert False, "invalid clusby"
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(zip([y, y_pred, X_behs, y_residual],
                                                 [f"obs. act. \n (subtract_psth={subtract_psth})",
                                                  f"pred. act. \n (subtract_psth={subtract_psth})",
                                                  var_behlist,
                                                  "residual act."],
                                                 [axes[0], axes[1], axes[2:-2], axes[-2]])):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)
        elif ri == 2:
            # plot behavior
            for bi in range(len(var_behlist)):
                ts_ = toshow[:, :, bi][t_sort]
                vmax = np.percentile(ts_, vmax_perc)
                vmin = np.percentile(ts_, vmin_perc)
                raster_plot(ts_, vmax, vmin, True, label[bi], ax[bi],
                            cmap=cmap,
                            aligned_tbins=aligned_tbins)
        elif ri == 3:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True,
                "residual act. (re-clustered)", axes[-1])

    plt.tight_layout()


"""
This script generates a plot to examine the (single-trial) fitting of a single neuron.
:X: behavior matrix of the shape [n_trials, n_timesteps, n_variables]. 
:y: true neural activity matrix of the shape [n_trials, n_timesteps] 
:ypred: predicted activity matrix of the shape [n_trials, n_timesteps] 
:var_name2idx: dictionary mapping feature names to their corresponding index of the 3-rd axis of the behavior matrix X. e.g.: {"choice": [0], "wheel": [1]}
:var_tasklist: *static* task variables used to form the task condition and compute the psth. e.g.: ["choice"]
:var_value2label: dictionary mapping values in X to their corresponding readable labels (only required for static task variables). e.g.: {"choice": {1.: "left", -1.: "right"}}
:var_behlist: *dynamic* behavior variables. e.g., ["wheel"]
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:algined_tbins: reference time steps to annotate in the plot. 
"""

def viz_single_cell(X, y, y_pred, var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[], clusby='y_pred', neuron_idx='', neuron_region='', method='',
                    save_path='figs', save_plot=False):
    
    if save_plot:
        nrows = 8
        plt.figure(figsize=(8, 2 * nrows))
        axes_psth = [plt.subplot(nrows, len(var_tasklist), k + 1) for k in range(len(var_tasklist))]
        axes_single = [plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)]
    else:
        axes_psth = None
        axes_single = None


    ### plot psth
    r2_psth, r2_trial = plot_psth(X, y, y_pred,
                                  var_tasklist=var_tasklist,
                                  var_name2idx=var_name2idx,
                                  var_value2label=var_value2label,
                                  aligned_tbins=aligned_tbins,
                                  axes=axes_psth, legend=True, neuron_idx=neuron_idx, neuron_region=neuron_region,
                                  save_plot=save_plot)

    ### plot the psth-subtracted activity
    if save_plot:
        plot_single_trial_activity(X, y, y_pred,
                                   var_name2idx,
                                   var_behlist,
                                   var_tasklist, subtract_psth=subtract_psth,
                                   aligned_tbins=aligned_tbins,
                                   clusby=clusby,
                                   axes=axes_single)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if save_plot:
        plt.savefig(os.path.join(save_path, f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2_trial:.2f}_{method}.png"))
        plt.tight_layout();

    return r2_psth, r2_trial
    

def viz_single_cell_unaligned(
    gt, pred, neuron_idx, neuron_region, method, save_path, 
    n_clus=8, n_neighbors=5, save_plot=False
):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    r2 = 0
    for _ in range(len(gt)):
        r2 += r2_score(gt, pred)
    r2 /= len(gt)

    if save_plot:
        y = gt - gt.mean(0)
        y_pred = pred - pred.mean(0)
        y_resid = y - y_pred

        clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                            affinity='nearest_neighbors',
                                            assign_labels='discretize',
                                            random_state=0)

        clustering = clustering.fit(y_pred)
        t_sort = np.argsort(clustering.labels_)
        
        vmin_perc, vmax_perc = 10, 90 
        vmax = np.percentile(y_pred, vmax_perc)
        vmin = np.percentile(y_pred, vmin_perc)
        
        toshow = [y, y_pred, y_resid]
        resid_vmax = np.percentile(toshow, vmax_perc)
        resid_vmin = np.percentile(toshow, vmin_perc)
        
        N = len(y)
        y_labels = ['obs.', 'pred.', 'resid.']

        fig, axes = plt.subplots(3, 1, figsize=(8, 7))
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im1 = axes[0].imshow(y[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im1, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        axes[0].set_title(f' R2: {r2:.3f}')
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im2 = axes[1].imshow(y_pred[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im2, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        norm = colors.TwoSlopeNorm(vmin=resid_vmin, vcenter=0, vmax=resid_vmax)
        im3 = axes[2].imshow(y_resid[t_sort], aspect='auto', cmap='bwr', norm=norm)
        cbar = plt.colorbar(im3, pad=0.02, shrink=.6)
        cbar.ax.tick_params(rotation=90)
        
        for i, ax in enumerate(axes):
            ax.set_ylabel(f"{y_labels[i]}"+f"\n(#trials={N})")
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
        
        plt.savefig(os.path.join(save_path, f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2:.2f}_{method}.png"))
        plt.tight_layout()

    return r2


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin - 1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
                    xy=(tbin - 1, N),
                    xytext=(tbin - 1, N + 10),
                    ha='center',
                    va='center',
                    arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}" + f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')


"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list
- return: nparray [T, N] or [T]
"""
def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)


def compute_all_psth(X, y, idxs_psth):
    uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
    psth_vs = {};
    for v in uni_vs:
        # compute separately for true y and predicted y
        _psth = compute_PSTH(X, y,
                             axis=idxs_psth, value=v)  # (T)
        psth_vs[tuple(v)] = _psth
    return psth_vs


"""
psth_xy/ psth_pred_xy: {tuple(x): (T) or (T,N)}
return a float or (N) array
"""
def compute_R2_psth(psth_xy, psth_pred_xy, clip=True):
    psth_xy_array = np.array([psth_xy[x] for x in psth_xy])
    psth_pred_xy_array = np.array([psth_pred_xy[x] for x in psth_xy])
    K, T = psth_xy_array.shape[:2]
    psth_xy_array = psth_xy_array.reshape((K * T, -1))
    psth_pred_xy_array = psth_pred_xy_array.reshape((K * T, -1))
    r2s = [r2_score(psth_xy_array[:, ni], psth_pred_xy_array[:, ni]) for ni in range(psth_xy_array.shape[1])]
    r2s = np.array(r2s)
    # # compute r2 along dim 0
    # r2s = [r2_score(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s, 0., 1.)
    # r2s = np.mean(r2s, 0)
    if len(r2s) == 1:
        r2s = r2s[0]
    return r2s


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
        
        
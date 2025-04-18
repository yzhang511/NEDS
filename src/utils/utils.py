import os
import time
import glob
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score as r2_score_sklearn
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(f"{PROJ_DIR}/data/test_eids.txt") as file:
    test_eids = file.read().splitlines()
    
def dummy_load(stop_event, dummy_size=60000, check_interval=1, device="cuda"):
    dummy_linear = torch.nn.Linear(dummy_size, dummy_size).to(device)
    with torch.no_grad():
        while not stop_event.is_set():
            x_ = dummy_linear(torch.rand(2048,dummy_size).to(device))
            time.sleep(check_interval)  # Adjust the sleep interval as needed

def set_seed(seed):
    print("Global seed set to {}.".format(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def plot_gt_pred(gt, pred, epoch=0,modality="behavior"):
    # plot Ground Truth and Prediction in the same figur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}, Mod: {}".format(epoch, 
                                             modality))
    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[],modality="behavior"):
    # Create one figure and axis for all plots
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for neuron in neuron_idx:
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[neuron_idx.index(neuron)]
        ax.plot(gt[:, neuron].cpu().numpy(), label="Ground Truth", color="blue")
        ax.plot(pred[:, neuron].cpu().numpy(), label="Prediction", color="red")
        ax.set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax.legend()
        # x label
        ax.set_xlabel("Time")
        # y label
        ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Mod: {}, Avg R2: {:.4f}".format(epoch, 
                                                            modality, 
                                                            np.mean(r2_values)))
    return fig

def plt_condition_avg_r2(gt, pred, epoch=0, neuron_idx=0, condition_idx=0, first_n=8, device="cpu"):
    _, unique, counts = np.unique(gt.cpu().numpy(), axis=0, return_inverse=True, return_counts=True)
    trial_idx = (unique == condition_idx)

    if trial_idx.sum() < first_n:
        first_n = trial_idx.sum()

    gt_condition = gt[trial_idx][0,:,neuron_idx]
    pred_condition = pred[trial_idx][:first_n,:,neuron_idx]
    
    # plot line of gt and pred in different colors
    r2 = r2_score(y_true=gt_condition, y_pred=torch.mean(pred_condition, axis=0), device=device)
    gt_condition = gt_condition.cpu().numpy()
    pred_condition = pred_condition.cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(gt_condition, label="Ground Truth", color="blue")
    # plot all pred trials, and show the range of the first_n trials through the shaded area
    ax.plot(np.mean(pred_condition, axis=0), label="Prediction", color="red")
    ax.fill_between(np.arange(pred_condition.shape[1]), np.min(pred_condition, axis=0), np.max(pred_condition, axis=0), color="red", alpha=0.2)

    ax.set_title("R2: {:.4f}".format(r2))
    ax.legend()
    # x label
    ax.set_xlabel("Time")
    # y label
    ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Neuron: {}, Condition: {}, Avg {} trials".format(epoch, neuron_idx, condition_idx, first_n))
    return fig


from scipy.special import gammaln
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

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}

    if "bps" in metrics:
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        bps_list = []
        for i in range(gt.shape[-1]): 
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    
    if "r2" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2s = [r2_score(y_true=gt[i].T[k], y_pred=pred[i].T[k], device=device) for k in range(len(gt[i].T))]
            r2_list.append(np.ma.masked_invalid(r2s).mean())
        r2 = np.mean(r2_list)
        results["r2"] = r2
        
    if "behave_r2" in metrics:
        r2_list = []
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        for i in range(gt.shape[0]):
           r2 = r2_score_sklearn(gt[i].flatten(), pred[i].flatten())
           r2_list.append(r2)
        mean_r2 = np.nanmean(r2_list)
        results["behave_r2"] = mean_r2
        
    if "rsquared" in metrics:
        r2 = 0
        for i in range(gt.shape[-1]):
            r2_list = []
            for j in range(gt.shape[0]):
                r2 = r2_score(y_true=gt[j,:,i], y_pred=pred[j,:,i], device=device) 
                r2_list.append(r2)
            r2 += np.mean(r2_list)
        results["rsquared"] = r2 / gt.shape[-1]
        
    if "mse" in metrics:
        mse = torch.mean((gt - pred) ** 2)
        results["mse"] = mse
        
    if "mae" in metrics:
        mae = torch.mean(torch.abs(gt - pred))
        results["mae"] = mae
        
    if "acc" in metrics:
        acc = accuracy_score(gt.cpu().numpy(), pred.cpu().detach().numpy())
        results["acc"] = acc
        
    return results


# plot the GT rate and the Prediction rate (after exp). TODO: add the spikes. unavailable now.
def plot_rate_and_spike(gt, pred, epoch=0, eps=1e-7):
    # find the most active neurons
    row_sum = np.sum(gt, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_top6 = pred[idxs_top6, :]
    gt_top6 = gt[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 6))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(-2, 3)
        # normalized TODO: change the var to be more serious
        y1 = (gt_top6[i, :] - np.mean(gt_top6[i, :])) / (np.sqrt(np.var(gt_top6[i, :])) + eps)
        y2 = (pred_top6[i, :] - np.mean(pred_top6[i, :])) / (np.sqrt(np.var(pred_top6[i, :])) + eps)
        ax.plot(y1, 'grey')
        ax.plot(y2, 'blue')
        ax.set_title(f"Neuron #{idxs_top6[i]}")

    fig.suptitle(f"Epoch: {epoch}")
    return fig


# plot the trail average rate and spike counts
def plot_avg_rate_and_spike(output, epoch=0):  # output is the list of batches (n_batch, 2, bs, seq_len, n_neurons)

    pred_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    gt_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    n_trials = 0

    for i, batch in enumerate(output):
        pred = batch[0].detach().cpu().numpy()
        gt = batch[1].detach().cpu().numpy()
        pred_avg = pred_avg + np.sum(pred, axis=0)
        gt_avg = gt_avg + np.sum(gt, axis=0)
        n_trials = n_trials + pred.shape[0]
    pred_avg = pred_avg.T / n_trials
    gt_avg = gt_avg.T / n_trials

    # find the most active neurons
    row_sum = np.sum(gt_avg, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_avg_top6 = pred_avg[idxs_top6, :]
    gt_avg_top6 = gt_avg[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(0, 2)
        y1 = pred_avg_top6[i, :]
        y2 = gt_avg_top6[i, :]
        ax.plot(y1, 'blue')
        ax.plot(y2, 'grey')
        ax.set_title(f'Neuron: #{idxs_top6[i]}')

    fig.suptitle(f"Epoch: {epoch},Avg from {n_trials} trials")
    return fig


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
              axes=None, legend=False, neuron_idx=0):
    if axes is None:
        nrows = 1; ncols = len(var_tasklist)
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
                    linewidth=3, alpha=0.3, label=f"{var_value2label[var][tuple(_x)]}")
            ax.plot(psth_pred,
                    color = plt.get_cmap('tab10')(_i),
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
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy)
    r2_single_trial = np.mean(compute_R2_main(y, y_pred, clip=False))
    axes[0].set_ylabel(f'Neuron: #{neuron_idx} \n PSTH R2: {r2_psth:.2f} \n Pred R2: {r2_single_trial:.2f}')    
    
    for ax in axes:
        # ax.axis('off')
        ax.spines[['right', 'top']].set_visible(False)
        # ax.set_frame_on(False)
        # ax.tick_params(bottom=False, left=False)
    plt.tight_layout()

    return {"psth_r2": r2_psth,
            "pred_r2": r2_single_trial}

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
        ncols = 1; nrows = 2+len(var_behlist)+1+1
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
        y = y -y_psth  # (K, T)
        y_pred = y-y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"
    y_residual = (y_pred - y)  # (K, T), residuals of prediction
    idxs_behavior = np.concatenate(([var_name2idx[var] for var in var_behlist])) if len(var_behlist)>0 else []
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
    clustering = SpectralClustering(n_clusters=n_clus,n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True, "residual act. (re-clustered)", axes[-1])

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
                    subtract_psth="task", aligned_tbins=[], clusby='y_pred', neuron_idx=0):
    nrows = 8
    plt.figure(figsize=(8, 2 * nrows))

    ### plot psth
    axes_psth = [plt.subplot(nrows, len(var_tasklist), k+1) for k in range(len(var_tasklist))]
    metrics = plot_psth(X, y, y_pred,
              var_tasklist=var_tasklist,
              var_name2idx=var_name2idx,
              var_value2label=var_value2label,
              aligned_tbins=aligned_tbins,
              axes=axes_psth, legend=True, neuron_idx=neuron_idx)

    ### plot the psth-subtracted activity
    axes_single = [plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)]
    plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth=subtract_psth,
                               aligned_tbins=aligned_tbins,
                               clusby=clusby,
                               axes=axes_single)

    fig_name = 'single_neuron' + str(neuron_idx)
    plt.tight_layout()
    return metrics
    # plt.show()

def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin-1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
            xy=(tbin-1, N),
            xytext=(tbin-1, N+10),
            ha='center',
            va='center',
            arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}"+f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
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
    psth_vs = {}
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
    # compute r2 along dim 0
    r2s = [r2_score_sklearn(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s,0.,1.)
    r2s = np.mean(r2s, 0)
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
    r2s = np.asarray([r2_score_sklearn(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
    
def prep_cond_matrix(test_dataset):
    b_list = []
    # choice
    choice = np.array(test_dataset['choice'])
    choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, 100))
    b_list.append(choice)
    # reward
    reward = np.array(test_dataset['reward'])
    reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, 100))
    b_list.append(reward)
    # block
    block = np.array(test_dataset['block'])
    block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, 100))
    b_list.append(block)
    # wheel
    wheel = np.array(test_dataset['wheel-speed'])
    b_list.append(wheel)
    behavior_set = np.stack(b_list,axis=-1)
    return behavior_set

var_name2idx = {'block':[2], 
                'choice': [0], 
                'reward': [1], 
                'wheel': [3],
                }

var_value2label = {'block': {(0.2,): "p(left)=0.2",
                            (0.5,): "p(left)=0.5",
                            (0.8,): "p(left)=0.8",},
                   'choice': {(-1.0,): "right",
                            (1.0,): "left"},
                   'reward': {(0.,): "no reward",
                            (1.,): "reward", } }

var_tasklist = ['block','choice','reward']


def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:,:,i] = (arr == _uni)
    return ret

def _std(arr):
    mean = np.mean(arr, axis=0) # (T, N)
    std = np.std(arr, axis=0) # (T, N)
    std = np.clip(std, 1e-8, None) # (T, N) 
    arr = (arr - mean) / std
    return arr, mean, std


def get_npy_files(log_dir, 
                  model_mode='mm',
                  num_sessions=1,
                  use_contrastive=True,
                  mixed_training=True,
                    ):
    """
    Get the npy files for each session
    """
    if model_mode == 'mm':
        model_mode = 'ap-wheel'
    elif model_mode == 'decoding':
        model_mode = 'ap'
    elif model_mode == 'encoding':
        model_mode = 'wheel'
    else:
        raise ValueError("Unknown model_mode")
    model_mode = f"inModal-{model_mode}"
    # get the npy files under sesNum-{num_sessions}/set-eval
    pattern = f"{log_dir}/sesNum-{num_sessions}/**/*{model_mode}*/**/*.npy"
    npy_files_ = glob.glob(pattern, recursive=True)
    npy_files = []
    test_eids_ = [ses[:5] for ses in test_eids]
    for npy_file in npy_files_:
        ses = npy_file.split('ses-')[1].split('/')[0]
        if ses in test_eids_:
            npy_files.append(npy_file)
    
    # filter the npy files
    if use_contrastive:
        npy_files = [f for f in npy_files if 'contrast-True' in f]
    else:
        npy_files = [f for f in npy_files if 'contrast-False' in f]
    if mixed_training:
        npy_files = [f for f in npy_files if 'mixedTraining-True' in f]
    else:
        npy_files = [f for f in npy_files if 'mixedTraining-False' in f]

    # get behav modal, decoding
    behav_modal = [f for f in npy_files if 'modal_behavior' in f]
    # remove bps in decoding
    behav_modal = [f for f in behav_modal if 'bps' not in f]

    spike_modal = [f for f in npy_files if 'modal_spike' in f]
    # remove r2 in spike
    spike_modal = [f for f in spike_modal if 'r2' not in f]

    npy_files = {
        'spike': spike_modal,
        'behavior': behav_modal
    }

    return npy_files

def return_behav_r2(npy_files, avail_beh = ['wheel-speed', 'whisker-motion-energy']):
    r2_list = []
    choice_acc_list = []
    block_acc_list = []
    choice_balanced_acc_list = []
    block_balanced_acc_list = []
    r2_dict = {}
    acc_dict = {}
    for npy_file in npy_files['behavior']:
        ses = npy_file.split('ses-')[1].split('/')[0]
        if 'acc.npy' in npy_file:
            decoding_data = np.load(npy_file, allow_pickle=True)
            choice_acc_list.append(decoding_data[0])
            block_acc_list.append(decoding_data[1])
            choice_balanced_acc_list.append(decoding_data[2])
            block_balanced_acc_list.append(decoding_data[3])
            print(f"session {ses} choice acc: {decoding_data[0]}, block acc: {decoding_data[1]} balanced choice acc: {decoding_data[2]}, balanced block acc: {decoding_data[3]}")
            acc_dict[ses] = {'choice_acc': decoding_data[0], 'block_acc': decoding_data[1], 'choice_balanced_acc': decoding_data[2], 'block_balanced_acc': decoding_data[3]}
        else:
            decoding_data = np.load(npy_file, allow_pickle=True)
            decoding_data = decoding_data.item()
            # only remain key with r2_trial
            decoding_data = {k: decoding_data[k] for k in decoding_data if 'r2_trial' in k}
            # only remain key with avail_beh
            decoding_data = {k: decoding_data[k] for k in decoding_data if any([beh in k for beh in avail_beh])}
            r2_list.append(decoding_data)
            print(f"session {ses} behavior decoding r2: {decoding_data}")
            r2_dict[ses] = decoding_data
    print("total {} sessions of behavior decoding".format(len(r2_list)))
    # return r2 for each session
    behav_result = {avail_beh[i]: [] for i in range(len(avail_beh))}
    for r2 in r2_list:
        for beh in avail_beh:
            behav_result[beh].append(np.mean(r2[f'{beh}_r2_trial']))
    behav_result['choice_acc'] = choice_acc_list
    behav_result['block_acc'] = block_acc_list
    behav_result['choice_balanced_acc'] = choice_balanced_acc_list
    behav_result['block_balanced_acc'] = block_balanced_acc_list
    # merge the r2 and acc dict based on the session
    assert len(r2_dict) == len(acc_dict), "r2 and acc dict should have the same length"
    all_ses = list(r2_dict.keys())
    all_dict = {}
    for ses in all_ses:
        all_dict[ses] = {acc_dict[ses]['choice_acc'], acc_dict[ses]['block_acc'], acc_dict[ses]['choice_balanced_acc'], acc_dict[ses]['block_balanced_acc'], r2_dict[ses]['wheel-speed_r2_trial'], r2_dict[ses]['whisker-motion-energy_r2_trial']}
        # save 5 decimal
        all_dict[ses] = {round(k, 5) for k in all_dict[ses]}
    if len(all_dict) == 0:
        print("No behavior decoding result found")
    return behav_result, all_dict

def return_spike_bps(npy_files):
    bps_list = []
    all_dict = {}
    for npy_file in npy_files['spike']:
        ses = npy_file.split('ses-')[1].split('/')[0]
        encoding_data = np.load(npy_file, allow_pickle=True)
        mean_bps = np.nanmean(encoding_data)
        bps_list.append(mean_bps)
        all_dict[ses] = mean_bps
        print(f"session {ses} spike encoding bps: {mean_bps}")
    print("total {} sessions of spike encoding".format(len(bps_list)))
    return bps_list, all_dict

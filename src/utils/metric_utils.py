# This file contains the implementation of the r2 score metric
from torcheval.metrics import R2Score
import torch
import torch.nn.functional as F
from scipy.special import gammaln
import numpy as np
from sklearn.metrics import r2_score as r2_score_sklearn

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def clip_contrastive_loss(similarity_matrix):
    """
    Compute CLIP's contrastive loss given a similarity matrix.
    The matrix contains cosine similarities of two sets of features.
    """
    labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    percent_correct = topk(similarity_matrix, labels, k=1)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)
    return (loss_i + loss_t) / 2, percent_correct

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
    
# get neuron metrics results
def compute_neuron_metrics(data_dict, metrics=["bps", "r2"], norm=False):
    results = {}
    gt, pred = data_dict["gt"], data_dict["pred"]
    print(f"gt shape: {gt.shape}, pred shape: {pred.shape}")
    # loop over the neurons
    if "bps" in metrics:
        num_neurons = gt.shape[-1]
        bps_list = []
        for i in range(num_neurons):
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        bps_list = np.array(bps_list)
        results["bps"] = bps_list

    if "ve" in metrics:
        # reshape the gt and pred
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        ven = compute_varexp(y_true=_gt, y_pred=_pred)
        results["ve"] = ven

    if "r2" in metrics:
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        r2 = r2_score_sklearn(y_true=_gt, y_pred=_pred, multioutput="raw_values")
        results["r2"] = r2
    return results
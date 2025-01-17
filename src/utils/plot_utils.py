import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
import imageio
from sklearn.metrics import r2_score 
from matplotlib.animation import PillowWriter
from sklearn.cluster import SpectralClustering
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec


def compute_psth(neural):
    psth = np.mean(neural, axis=0)
    return psth

def plot_combined_neuron_scatter(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        metric=['bps', 'r2'],
        title="Neuron Scatter Plot",
    ):
    """
    Plot a scatter plot of neuron metrics for two models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, m in enumerate(metric):
        plot_neuron_scatter(
            x_dict, 
            y_dict, 
            x_name=x_name,
            y_name=y_name,
            metric=m,
            ax=axes[i]
        )
    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes

def plot_neuron_scatter(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        metric='bps',
        ax=None,
    ):
    """
    Plot a scatter plot of neuron metrics for two models.
    
    Parameters
    ----------
    x_dict : dict
        A dictionary of neuron metrics for the base model.
    y_dict : dict
        A dictionary of neuron metrics for the new model.
    x_name : str, optional
        The name of the base model for the x-axis.
    y_name : str, optional
        The name of the new model for the y-axis.
    """
    base_neuron_values = x_dict[metric]
    new_neuron_values = y_dict[metric]
    assert len(base_neuron_values) == len(new_neuron_values), "Neuron counts must match."
    N = len(base_neuron_values)
    # return fig and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    # calculate mean r2 and bps
    x_metric = np.mean(base_neuron_values)
    y_metric = np.mean(new_neuron_values)
    ax.scatter(base_neuron_values, new_neuron_values, alpha=0.5)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{metric} {x_name}: {x_metric:.2f} {y_name}: {y_metric:.2f}")
    # set x from min to max
    min_x = min(min(base_neuron_values), min(new_neuron_values))
    max_x = max(max(base_neuron_values), max(new_neuron_values))
    ax.set_xlim(min_x, max_x)
    # set y from min to max
    min_y = min(min(base_neuron_values), min(new_neuron_values))
    max_y = max(max(base_neuron_values), max(new_neuron_values))
    ax.set_ylim(min_y, max_y)
    # plot diagonal line, slope=1
    ax.plot([min_x, max_x], [min_x, max_x], color='red', linestyle='--')
    plt.tight_layout()
    return fig, ax
    
def plot_neuron_raster(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        neuron_idx=None,
        subtract='global',
        n_clus=8,
        n_neighbors=5,
        ax=None,
    ):
    """
    Plot a raster plot of neuron metrics for two models.
    
    Parameters
    ----------
    x_dict : dict
        A dictionary of neuron metrics for the base model.
    y_dict : dict
        A dictionary of neuron metrics for the new model.
    x_name : str, optional
        The name of the base model for the x-axis.
    y_name : str, optional
        The name of the new model for the y-axis.
    """
    gt, x_pred, y_pred = x_dict['gt'], x_dict['pred'], y_dict['pred']
    N = len(gt)
    if neuron_idx is None:
        # select the most active neurons
        fr = np.mean(gt, axis=(0, 1))
        neuron_idx = np.argmax(fr)

    gt, x_pred, y_pred = gt[:, :, neuron_idx], x_pred[:, :, neuron_idx], y_pred[:, :, neuron_idx]
    # neuron r2
    x_r2 = r2_score(gt.flatten(), x_pred.flatten())
    y_r2 = r2_score(gt.flatten(), y_pred.flatten())
    if subtract == 'global':
        gt = gt - gt.mean(0)
        x_pred = x_pred - x_pred.mean(0)
        y_pred = y_pred - y_pred.mean(0)
    else:
        raise ValueError("Invalid subtraction method.")

    clustering = SpectralClustering(
        n_clusters=n_clus, 
        n_neighbors=n_neighbors,
        affinity='nearest_neighbors',
        assign_labels='discretize',
        random_state=0
    )
    clustering = clustering.fit(y_pred)
    t_sort = np.argsort(clustering.labels_)
    vmin_perc, vmax_perc = 10, 90 
    vmax = np.percentile(y_pred, vmax_perc)
    vmin = np.percentile(y_pred, vmin_perc)

    # sort gt, x_pred, y_pred
    gt, x_pred, y_pred = gt[t_sort], x_pred[t_sort], y_pred[t_sort]
    # plot raster, use imshow. [Trial, Time]
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    else:
        fig = ax[0].get_figure()
        axes = ax
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # plot ground truth
    # axes[0].set_title(f"Neuron {neuron_idx} {x_name} R2: {x_r2:.2f} {y_name} R2: {y_r2:.2f}")
    axes[0].imshow(gt, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("GT \n"
                       f"subtract_psth: {subtract}\n"
                       "Trial")
    fig.colorbar(axes[0].imshow(gt, aspect='auto', cmap='bwr', origin='lower'), ax=axes[0])
    
    # plot x_pred
    axes[1].imshow(x_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[1].set_title(f"{x_name}")
    axes[1].set_xlabel("Time")
    fig.colorbar(axes[1].imshow(x_pred, aspect='auto', cmap='bwr', origin='lower'), ax=axes[1])

    axes[2].imshow(y_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[2].set_title(f"{y_name}")
    axes[2].set_xlabel("Time")
    fig.colorbar(axes[2].imshow(y_pred, aspect='auto', cmap='bwr', origin='lower'), ax=axes[2])
    plt.tight_layout()
    return fig, axes

def plot_psth(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        neuron_idx=None,
        ax=None,
    ):
    """
    Plot a PSTH plot of neuron metrics for two models.
    
    Parameters
    ----------
    x_dict : dict
        A dictionary of neuron metrics for the base model.
    y_dict : dict
        A dictionary of neuron metrics for the new model.
    x_name : str, optional
        The name of the base model for the x-axis.
    y_name : str, optional
        The name of the new model for the y-axis.
    """
    gt, x_pred, y_pred = x_dict['gt'], x_dict['pred'], y_dict['pred']
    N = len(gt)
    if neuron_idx is None:
        # select the most active neurons
        fr = np.mean(gt, axis=(0, 1))
        neuron_idx = np.argmax(fr)

    gt, x_pred, y_pred = gt[:, :, neuron_idx], x_pred[:, :, neuron_idx], y_pred[:, :, neuron_idx]
    num_trials, num_time = gt.shape
    # compute psth
    gt_psth = compute_psth(gt)
    x_pred_psth = compute_psth(x_pred)
    y_pred_psth = compute_psth(y_pred)
    # normalize psth to [0, 1]
    gt_psth = (gt_psth - min(gt_psth)) / (max(gt_psth) - min(gt_psth))
    x_pred_psth = (x_pred_psth - min(x_pred_psth)) / (max(x_pred_psth) - min(x_pred_psth))
    y_pred_psth = (y_pred_psth - min(y_pred_psth)) / (max(y_pred_psth) - min(y_pred_psth))
    # x psth r2
    x_r2 = r2_score(gt_psth, x_pred_psth)
    # y psth r2
    y_r2 = r2_score(gt_psth, y_pred_psth)
    # plot psth
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    ax.plot(gt_psth, label='GT')
    ax.plot(x_pred_psth, label=x_name)
    ax.plot(y_pred_psth, label=y_name)
    ax.set_title(f"PSTH Neuron {neuron_idx}")
    ax.set_xlabel("Time")
    ax.set_ylabel(
        f"{x_name} PSTH R2: {x_r2:.2f} \n"
        f"{y_name} PSTH R2: {y_r2:.2f} \n"
        f"Normalized Firing Rate"
    )
    ax.legend()
    plt.tight_layout()
    return fig, ax    

def plot_combined_neuron_psth_raster(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        neuron_idx=None,
        subtract='global',
        n_clus=8,
        n_neighbors=5,
        title="Neuron PSTH and Raster Plot",
    ):
    """
    Plot a combined raster and PSTH plot of neuron metrics for two models.
    """
    # Create the main figure with GridSpec
    fig = plt.figure(figsize=(8, 12))
    # 2 rows, 2 column
    # top row for PSTH
    # bottom row for raster
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 0.14], hspace=0.2)

    # top subplot for PSTH
    ax_psth = fig.add_subplot(gs[0, 0])
    plot_psth(
        x_dict, 
        y_dict, 
        x_name=x_name, 
        y_name=y_name, 
        neuron_idx=neuron_idx, 
        ax=ax_psth
    )

    # bottom subplot for raster
    gs_raster = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, :], hspace=0.4)
    ax_raster = [fig.add_subplot(gs_raster[i, 0]) for i in range(3)]
    plot_neuron_raster(
        x_dict, 
        y_dict, 
        x_name=x_name, 
        y_name=y_name, 
        neuron_idx=neuron_idx, 
        subtract=subtract,
        n_clus=n_clus,
        n_neighbors=n_neighbors,
        ax=ax_raster
    )
    # title
    fig.suptitle(title)
    plt.tight_layout()
    return fig

def plot_neuron_raster_single_model(
        x_dict, 
        neuron_idx=None,
        n_clus=8,
        n_neighbors=5,
    ):
    """
    Plot a raster plot of neuron metrics for two models.
    
    Parameters
    ----------
    x_dict : dict
        A dictionary of neuron metrics for the base model.
    x_name : str, optional
        The name of the base model for the x-axis.
    """
    gt, x_pred= x_dict['gt'], x_dict['pred']
    N = len(gt)
    if neuron_idx is None:
        # select the most active neurons
        fr = np.mean(gt, axis=(0, 1))
        neuron_idx = np.argmax(fr)

    gt, x_pred= gt[:, :, neuron_idx], x_pred[:, :, neuron_idx]
    fig, ax = viz_single_cell_unaligned(
        gt=gt,
        pred=x_pred,
        n_clus=n_clus,
        n_neighbors=n_neighbors,
    )
    return fig, ax

def viz_single_cell_unaligned(
    gt, 
    pred, 
    n_clus=8, 
    n_neighbors=5, 
):

    r2 = 0
    for _ in range(len(gt)):
        r2 += r2_score(gt, pred)
    r2 /= len(gt)

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
    
    plt.tight_layout()

    return fig, axes
# Example usage:
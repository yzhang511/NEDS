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
import pandas as pd
import seaborn as sns

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
        show_colorbar=True,
        show_info=True,
        show_xticks=True,
        show_model_name=True,
        text_size=16,
        num_trials=16,
        line_width=3,
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
    gt, x_pred, y_pred = x_dict['gt'][:num_trials], x_dict['pred'][:num_trials], y_dict['pred'][:num_trials]
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
    im1 = axes[0].imshow(gt, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[0].set_title("Ground Truth") if show_model_name else None
    axes[0].set_xlabel("Time (s)") if show_xticks else None
    axes[0].set_ylabel("GT \n"
                       f"subtract_psth: {subtract}\n"
                       "Trial") if show_info else None
    # set yticks
    # set 0 point be N, and end point be 0
    axes[0].set_yticks([0, N-1])
    axes[0].set_yticklabels([N-1, 0])
    # set xticks else no xticks
    axes[0].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[0].set_xticks([])
    axes[0].set_xticklabels([0,'',2.0]) if show_xticks else axes[0].set_xticklabels([])
    fig.colorbar(im1, ax=axes[0]) if show_colorbar else None
    
    # plot x_pred
    im2 = axes[1].imshow(x_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[1].set_title(f"{x_name}") if show_model_name else None
    axes[1].set_xlabel("Time (s)") if show_xticks else None
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    # set xticks
    axes[1].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[1].set_xticks([])
    axes[1].set_xticklabels([0,'',2.0]) if show_xticks else axes[1].set_xticklabels([])    
    fig.colorbar(im2, ax=axes[1]) if show_colorbar else None
    
    # plot y_pred
    im3 = axes[2].imshow(y_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[2].set_title(f"{y_name}") if show_model_name else None
    axes[2].set_xlabel("Time (s)") if show_xticks else None
    axes[2].set_yticks([])
    axes[2].set_yticklabels([])
    # set xticks
    axes[2].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[2].set_xticks([])
    axes[2].set_xticklabels([0,'',2.0]) if show_xticks else axes[2].set_xticklabels([])
    fig.colorbar(im3, ax=axes[2]) if show_colorbar else None
    # set parameters for all axes
    for ax in axes:
        ax.set_title(ax.get_title(), pad=10, size=text_size)
        ax.set_xlabel(ax.get_xlabel(), labelpad=-15, size=text_size)
        ax.set_ylabel(ax.get_ylabel(), labelpad=10, size=text_size)
        ax.xaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
        ax.yaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
        # change all spines
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(line_width)
    # # set title for the figure
    fig.suptitle(f"Neuron {neuron_idx} {x_name} R2: {x_r2:.2f} {y_name} R2: {y_r2:.2f}") if show_info else None

    plt.tight_layout()
    return fig, axes

def plot_multi_neuron_raster(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        neuron_list=None,
        subtract='global',
        n_clus=8,
        n_neighbors=5,
        title="Neuron Raster Plot",
        text_size=16,
        num_trials=16,
        show_neuron_idx=True,
    ):
    """
    Plot a raster plot of neuron metrics for two models.
    """
    if neuron_list is None:
        # select the most active neurons
        fr = np.mean(x_dict['gt'], axis=(0, 1))
        neuron_list = np.argsort(fr)[::-1][:5]
    neuron_list = np.array(neuron_list)
    num_neurons = len(neuron_list)
    # call plot_neuron_raster for each neuron
    fig, axes = plt.subplots(num_neurons, 3, figsize=(18, 2*num_neurons))
    for i, neuron_idx in enumerate(neuron_list):
        fig_neuron, ax_neuron = plot_neuron_raster(
            x_dict, 
            y_dict, 
            x_name=x_name, 
            y_name=y_name, 
            neuron_idx=neuron_idx, 
            subtract=subtract,
            n_clus=n_clus,
            n_neighbors=n_neighbors,
            ax=axes[i],
            show_colorbar=False,
            show_info=False,
            show_xticks=True if i == num_neurons - 1 else False,
            show_model_name=True if i == 0 else False,
            text_size=text_size,
            num_trials=num_trials,
        )
        # move the ax to the right side of the figure
        ax_neuron[-1].yaxis.set_label_position("right")
        # ax label becomes neuron idx
        ax_neuron[-1].set_ylabel(f"Neuron {i}", size=text_size) if show_neuron_idx else None
    # set overall y label for the figure
    fig.text(
        0.015, 
        0.5, 
        'Trial Index', 
        ha='center', 
        va='center', 
        rotation='vertical',
        size=text_size
    )
    fig.subplots_adjust(wspace=0.15, left=0.06)
    return fig, axes

def plot_psth(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        neuron_idx=None,
        ax=None,
        show_info=True,
        show_xticks=True,
        show_yticks=True,
        text_size=16,
        num_trials=16,
        line_width=3,
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

    gt, x_pred, y_pred = gt[:num_trials, :, neuron_idx], x_pred[:num_trials, :, neuron_idx], y_pred[:num_trials, :, neuron_idx]
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
        fig, ax = plt.subplots(figsize=(4, 4), sharex=True)
    else:
        fig = ax.get_figure()
    ax.plot(gt_psth, label='GT', color='black')
    ax.plot(x_pred_psth, label=x_name)
    ax.plot(y_pred_psth, label=y_name, linestyle='--')
    ax.set_title(f"PSTH Neuron {neuron_idx}")
    ax.set_xlabel("Time") if show_info else None
    ax.set_ylabel(
        f"{x_name} PSTH R2: {x_r2:.2f} \n"
        f"{y_name} PSTH R2: {y_r2:.2f} \n"
        f"Normalized Firing Rate"
    ) if show_info else None
    # x ticks
    ax.set_xticks([0, num_time-1]) if show_xticks else ax.set_xticks([])
    ax.set_xticklabels([0, 2.0]) if show_xticks else ax.set_xticklabels([])
    # y ticks
    ax.set_yticks([0, 1]) if show_yticks else ax.set_yticks([])
    ax.set_yticklabels([0, 1]) if show_yticks else ax.set_yticklabels([])
    ax.legend() if show_info else None
    # set top spine invisible
    ax.spines['top'].set_visible(False)
    # set right spine invisible
    ax.spines['right'].set_visible(False)
    # set left spine invisible
    ax.spines['left'].set_visible(show_yticks)
    # set bottom spine invisible
    ax.spines['bottom'].set_visible(show_xticks)
    # set line width for all lines
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    # set plot line width
    for line in ax.lines:
        line.set_linewidth(line_width)
    # set size for all labels
    ax.set_title(ax.get_title(), size=text_size)
    ax.set_xlabel(ax.get_xlabel(), size=text_size)
    ax.set_ylabel(ax.get_ylabel(), size=text_size)
    ax.xaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
    ax.yaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
    plt.tight_layout()
    return fig, ax    

def plot_multi_neuron_psth(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        neuron_list=None,
        title="Neuron PSTH Plot",
        text_size=16,
        num_columns=3,
        num_rows=2,
        num_trials=16,
    ):
    """
    Plot a PSTH plot of neuron metrics for two models.
    """
    if neuron_list is None:
        # select the most active neurons
        fr = np.mean(x_dict['gt'], axis=(0, 1))
        neuron_list = np.argsort(fr)[::-1][:5]
    neuron_list = np.array(neuron_list)
    num_neurons = len(neuron_list)
    # call plot_psth for each neuron
    fig, axes = plt.subplots(
        num_rows, 
        num_columns, 
        figsize=(4*num_columns, 3*num_rows), 
        # sharex=True,
        # sharey=False
    )
    for i, neuron_idx in enumerate(neuron_list):
        row_idx = i // num_columns
        col_idx = i % num_columns
        # print(f"Neuron {i} at row {row_idx} col {col_idx}")
        ax = axes[row_idx, col_idx]
        fig_neuron, ax_neuron = plot_psth(
            x_dict, 
            y_dict, 
            x_name=x_name, 
            y_name=y_name, 
            neuron_idx=neuron_idx, 
            ax=ax,
            show_info=False,
            text_size=text_size,
            num_trials=num_trials,
            show_xticks=True if row_idx == num_rows - 1 else False,
            show_yticks=True if col_idx == 0 else False,
        )
        ax_neuron.set_title(f"Neuron {i+1}", size=text_size)
    # set overall y label for the figure
    fig.text(
        0.015, 
        0.5, 
        'Normalized Firing Rate', 
        ha='center', 
        va='center', 
        rotation='vertical',
        size=text_size
    )
    # set legend label Ground Truth, x_name, y_name; in color black, blue, red
    fig.text(
        0.5, 
        0.015, 
        'Time (s)', 
        ha='center', 
        va='center',
        size=text_size
    )
    # get the default color for three different lines
    gt_color = axes[0, 0].lines[0].get_color()
    x_color = axes[0, 0].lines[1].get_color()
    y_color = axes[0, 0].lines[2].get_color()
    # set text for each Ground Truth, x_name, y_name
    fig.text(
        0.81, 
        0.96, 
        'Ground Truth', 
        ha='center', 
        va='center',
        size=text_size,
        color=gt_color
    )
    
    fig.text(
        0.89, 
        0.96, 
        x_name, 
        ha='center', 
        va='center',
        size=text_size,
        color=x_color
    )
    fig.text(
        0.96, 
        0.96, 
        y_name, 
        ha='center', 
        va='center',
        size=text_size,
        color=y_color
    )

    # give some space for Time (s) label
    fig.subplots_adjust(
        wspace=0.15, 
        left=0.06,
        bottom=0.13,
        top=0.88
    )
    return fig, axes

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

def plot_behav_trial(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        trial_idx=None,
        ax=None,
        show_info=True,
        show_xticks=True,
        show_yticks=True,
        text_size=16,
        line_width=3,
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
    if trial_idx is None:
        trial_idx=0
    gt, x_pred, y_pred = gt[trial_idx] - gt.mean(0), x_pred[trial_idx] - x_pred.mean(0), y_pred[trial_idx] - y_pred.mean(0)
    num_time = len(gt)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), sharex=True)
    else:
        fig = ax.get_figure()
    ax.plot(gt, label='GT', color='black')
    ax.plot(x_pred, label=x_name)
    ax.plot(y_pred, label=y_name, linestyle='--')
    ax.set_title(f"PSTH Neuron {trial_idx}")
    ax.set_xlabel("Time") if show_info else None
    ax.set_ylabel(
        f"{x_name} PSTH R2: {x_r2:.2f} \n"
        f"{y_name} PSTH R2: {y_r2:.2f} \n"
        f"Normalized Firing Rate"
    ) if show_info else None
    # x ticks
    ax.set_xticks([0, num_time-1]) if show_xticks else ax.set_xticks([])
    ax.set_xticklabels([0, 2.0]) if show_xticks else ax.set_xticklabels([])
    # y ticks
    ax.set_yticks([0, 1]) if show_yticks else ax.set_yticks([])
    ax.set_yticklabels([0, 1]) if show_yticks else ax.set_yticklabels([])
    ax.legend() if show_info else None
    # set top spine invisible
    ax.spines['top'].set_visible(False)
    # set right spine invisible
    ax.spines['right'].set_visible(False)
    # set left spine invisible
    ax.spines['left'].set_visible(show_yticks)
    # set bottom spine invisible
    ax.spines['bottom'].set_visible(show_xticks)
    # set line width for all lines
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    # set plot line width
    for line in ax.lines:
        line.set_linewidth(line_width)
    # set size for all labels
    ax.set_title(ax.get_title(), size=text_size)
    ax.set_xlabel(ax.get_xlabel(), size=text_size)
    ax.set_ylabel(ax.get_ylabel(), size=text_size)
    ax.xaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
    ax.yaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
    plt.tight_layout()
    return fig, ax 

def plot_multi_trial(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        trial_list=None,
        title="Neuron PSTH Plot",
        text_size=16,
        num_columns=3,
        num_rows=2,
        num_trials=16,
        behav_name='Behavior',
    ):
    """
    Plot a PSTH plot of neuron metrics for two models.
    """
    if trial_list is None:
        trial_list = np.arange(5)
    trial_list = np.array(trial_list)
    # call plot_psth for each neuron
    fig, axes = plt.subplots(
        num_rows, 
        num_columns, 
        figsize=(4*num_columns, 3*num_rows), 
        # sharex=True,
        # sharey=False
    )
    for i, trial_idx in enumerate(trial_list):
        row_idx = i // num_columns
        col_idx = i % num_columns
        # print(f"Neuron {i} at row {row_idx} col {col_idx}")
        ax = axes[row_idx, col_idx]
        fig_neuron, ax_neuron = plot_behav_trial(
            x_dict, 
            y_dict, 
            x_name=x_name, 
            y_name=y_name, 
            trial_idx=trial_idx, 
            ax=ax,
            show_info=False,
            text_size=text_size,
            show_xticks=True if row_idx == num_rows - 1 else False,
            show_yticks=True if col_idx == 0 else False,
        )
        ax_neuron.set_title(f"Trial {i+1}", size=text_size)
    # set overall y label for the figure
    fig.text(
        0.015, 
        0.5, 
        behav_name, 
        ha='center', 
        va='center', 
        rotation='vertical',
        size=text_size
    )
    # set legend label Ground Truth, x_name, y_name; in color black, blue, red
    fig.text(
        0.5, 
        0.015, 
        'Time (s)', 
        ha='center', 
        va='center',
        size=text_size
    )
    # get the default color for three different lines
    gt_color = axes[0, 0].lines[0].get_color()
    x_color = axes[0, 0].lines[1].get_color()
    y_color = axes[0, 0].lines[2].get_color()
    # set text for each Ground Truth, x_name, y_name
    fig.text(
        0.81, 
        0.96, 
        'Ground Truth', 
        ha='center', 
        va='center',
        size=text_size,
        color=gt_color
    )
    
    fig.text(
        0.89, 
        0.96, 
        x_name, 
        ha='center', 
        va='center',
        size=text_size,
        color=x_color
    )
    fig.text(
        0.96, 
        0.96, 
        y_name, 
        ha='center', 
        va='center',
        size=text_size,
        color=y_color
    )

    # give some space for Time (s) label
    fig.subplots_adjust(
        wspace=0.15, 
        left=0.06,
        bottom=0.13,
        top=0.88
    )
    return fig, axes

def plot_behav_raster(
        x_dict, 
        y_dict, 
        x_name='Base Model',
        y_name='New Model',
        subtract='global',
        n_clus=8,
        n_neighbors=5,
        ax=None,
        show_colorbar=True,
        show_info=True,
        show_xticks=True,
        show_model_name=True,
        text_size=16,
        num_trials=16,
        line_width=3,
        behav_name='Behavior',
        if_plot=True,
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
    gt, x_pred, y_pred = x_dict['gt'][:num_trials], x_dict['pred'][:num_trials], y_dict['pred'][:num_trials]
    N = len(gt)

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
    im1 = axes[0].imshow(gt, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[0].set_title("Ground Truth") if show_model_name else None
    axes[0].set_xlabel("Time (s)") if show_xticks else None
    axes[0].set_ylabel("GT \n"
                       f"subtract_psth: {subtract}\n"
                       "Trial") if show_info else None
    # set yticks
    # set 0 point be N, and end point be 0
    axes[0].set_yticks([0, N-1])
    axes[0].set_yticklabels([N-1, 0])
    # set xticks else no xticks
    axes[0].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[0].set_xticks([])
    axes[0].set_xticklabels([0,'',2.0]) if show_xticks else axes[0].set_xticklabels([])
    fig.colorbar(im1, ax=axes[0]) if show_colorbar else None
    
    # plot x_pred
    im2 = axes[1].imshow(x_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[1].set_title(f"{x_name}") if show_model_name else None
    axes[1].set_xlabel("Time (s)") if show_xticks else None
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    # set xticks
    axes[1].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[1].set_xticks([])
    axes[1].set_xticklabels([0,'',2.0]) if show_xticks else axes[1].set_xticklabels([])    
    fig.colorbar(im2, ax=axes[1]) if show_colorbar else None
    
    # plot y_pred
    im3 = axes[2].imshow(y_pred, aspect='auto', cmap='bwr', origin='lower', norm=norm)
    axes[2].set_title(f"{y_name}") if show_model_name else None
    axes[2].set_xlabel("Time (s)") if show_xticks else None
    axes[2].set_yticks([])
    axes[2].set_yticklabels([])
    # set xticks
    axes[2].set_xticks([0, (len(gt[0])-1)//2,len(gt[0])-1]) if show_xticks else axes[2].set_xticks([])
    axes[2].set_xticklabels([0,'',2.0]) if show_xticks else axes[2].set_xticklabels([])
    fig.colorbar(im3, ax=axes[2]) if show_colorbar else None
    # set parameters for all axes
    for ax in axes:
        ax.set_title(ax.get_title(), pad=10, size=text_size)
        ax.set_xlabel(ax.get_xlabel(), labelpad=-15, size=text_size)
        ax.set_ylabel(ax.get_ylabel(), labelpad=10, size=text_size)
        ax.xaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
        ax.yaxis.set_tick_params(labelsize=text_size, width=line_width,length=10)
        # change all spines
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(line_width)
    # # set title for the figure
    fig.suptitle(f"{behav_name} {x_name} R2: {x_r2:.2f} {y_name} R2: {y_r2:.2f}") if show_info else None
    plt.tight_layout()
    fig.text(
        0.015, 
        0.5, 
        'Trial Index', 
        ha='center', 
        va='center', 
        rotation='vertical',
        size=text_size
    ) if if_plot else None
    fig.subplots_adjust(wspace=0.15, left=0.06)
    return fig, axes

def plot_multi_behav_raster(
        x_dict,
        y_dict,
        x_name='Base Model',
        y_name='New Model',
        subtract='global',
        n_clus=8,
        n_neighbors=5,
        title="Neuron Raster Plot",
        text_size=16,
        num_trials=16,
    ):
    """
    Plot a raster plot of neuron metrics for two models.
    """
    assert len(x_dict) == len(y_dict), "The number of behavs should be the same."
    num_behavs = len(x_dict)
    title_list = list(x_dict.keys())
    # call plot_neuron_raster for each neuron
    fig, axes = plt.subplots(num_behavs, 3, figsize=(18, 2*num_behavs + 1))
    # ensure the axes is a 2D array
    for behav_idx in range(num_behavs):
        fig_behav, ax_behav = plot_behav_raster(
            x_dict[title_list[behav_idx]],
            y_dict[title_list[behav_idx]],
            x_name=x_name, 
            y_name=y_name, 
            behav_name=title_list[behav_idx],
            subtract=subtract,
            n_clus=n_clus,
            n_neighbors=n_neighbors,
            ax=axes[behav_idx] if num_behavs > 1 else axes,
            show_colorbar=False,
            show_info=False,
            show_xticks=True if behav_idx == num_behavs - 1 else False,
            show_model_name=True if behav_idx == 0 else False,
            text_size=text_size,
            num_trials=num_trials,
        )
        # move the ax to the right side of the figure
        ax_behav[-1].yaxis.set_label_position("right")
        # ax label becomes neuron idx
        ax_behav[-1].set_ylabel(title_list[behav_idx], size=text_size)

    fig_behav.text(
        0.015, 
        0.5, 
        'Trial Index', 
        ha='center', 
        va='center', 
        rotation='vertical',
        size=text_size
    )
    fig.subplots_adjust(wspace=0.15, left=0.06)
    return fig, axes

def plot_encoding_boxplot(
        csv_path,
        model_list,
        model_name_list,
        color_list,
        text_size=16,
        line_width=3,
        title="Encoding",
):
    """
    Plot a barplot of the encoding results.
    """
    df = pd.read_csv(csv_path)
    model_res = {}
    print(color_list)
    for model in model_list:
        model = model
        # select columns that contain the model name
        model_df = df.filter(like=model, axis=1)
        # skip the first row of the model_df, and convert columns to list
        values = model_df.iloc[1:].values.flatten()
        values = values.astype(float)
        model_res[model] = values
    # plot the boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    # make the box plot median line be black
    ax.boxplot(
        model_res.values(),
        medianprops=dict(color='black')
    )
    # makes bar plot of the mean of each model
    # align with the boxplot
    x = np.arange(1, len(model_name_list) + 1)
    y = [np.mean(model_res[model]) for model in model_list]
    ax.bar(x, y, color=color_list)
    ax.set_xticks(x)
    ax.set_xticklabels(model_name_list)
    # set the mean value on the bottom of the bar
    for i, v in enumerate(y):
        ax.text(
            x=i + 1, 
            y=0, 
            s=f"{v:.2f}", 
            color='white', 
            ha='center', 
            va='bottom',
            size=text_size
        )
    ax.set_ylabel("Bits per spike")
    ax.set_title(title)
    # set size for all labels
    ax.set_title(ax.get_title(), size=text_size)
    ax.set_xlabel(ax.get_xlabel(), size=text_size)
    ax.set_ylabel(ax.get_ylabel(), size=text_size)
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)
    # set line width for all lines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(line_width)
    # set the boxplot line width
    for line in ax.lines:
        line.set_linewidth(line_width)
    plt.tight_layout()
    return fig, ax

# Example usage:
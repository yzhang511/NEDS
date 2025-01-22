import numpy as np
import os
from utils.metric_utils import (
    compute_neuron_metrics
)
from utils.plot_utils import (
    plot_multi_neuron_psth,
    plot_neuron_raster,
    plot_multi_neuron_raster,
    plot_multi_trial,
    plot_behav_raster
)
mm_dir = 'results/sesNum-1_ses-d23a4_set-eval_inModal-spike-choice-block-wheel-whisker_outModal-spike-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_spike'
encoding_dir = 'results/sesNum-1_ses-d23a4_set-eval_inModal-choice-block-wheel-whisker_outModal-spike_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_spike'
mm_data_path = os.path.join(mm_dir, 'data.npy')
encoding_data_path = os.path.join(encoding_dir, 'data.npy')

mm_whisker_data_path = '/scratch/yl6624/Project/multi_modal_foundation_model/results/sesNum-1_ses-db4df_set-eval_inModal-spike-choice-block-wheel-whisker_outModal-spike-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_behavior/data.npy'
decoding_whisker_data_path = '/scratch/yl6624/Project/multi_modal_foundation_model/results/sesNum-1_ses-db4df_set-eval_inModal-spike_outModal-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_behavior/data.npy'
mm_whisker_data = np.load(mm_whisker_data_path, allow_pickle=True).item()['whisker']
decoding_whisker_data = np.load(decoding_whisker_data_path, allow_pickle=True).item()['whisker']
fig, ax = plot_behav_raster(
    x_dict=decoding_whisker_data,
    y_dict=mm_whisker_data,
    x_name="Unimodal",
    y_name="Multimodal",
    behav_name="Whisker Motion Energy",
    num_trials=21,
    text_size=20,
    show_colorbar=False,
    show_info=True
)
fig.savefig("whisker_raster.png")

fig, ax = plot_multi_trial(
    x_dict=decoding_whisker_data,
    y_dict=mm_whisker_data,
    x_name="Unimodal",
    y_name="Multimodal",
    trial_list=[
        0,1,2,3,8,10,
        4,5,6,7,9,11
    ],
    text_size=20,
    num_rows=2,
    num_columns=6,
    behav_name="Whisker Motion Energy"
)
fig.savefig("whisker.png")
print("whisker done")
# load data
mm_wheel_data_path = '/scratch/yl6624/Project/multi_modal_foundation_model/results/sesNum-1_ses-db4df_set-eval_inModal-spike-choice-block-wheel-whisker_outModal-spike-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_behavior/data.npy'
decoding_wheel_data_path = '/scratch/yl6624/Project/multi_modal_foundation_model/results/sesNum-1_ses-db4df_set-eval_inModal-spike_outModal-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_behavior/data.npy'
mm_wheel_data = np.load(mm_wheel_data_path, allow_pickle=True).item()['wheel']
decoding_wheel_data = np.load(decoding_wheel_data_path, allow_pickle=True).item()['wheel']

fig, ax = plot_multi_trial(
    x_dict=decoding_wheel_data,
    y_dict=mm_wheel_data,
    x_name="Unimodal",
    y_name="Multimodal",
    trial_list=[
        0,13,2,3,8,10,
        4,14,6,7,9,11
    ],
    text_size=20,
    num_rows=2,
    num_columns=6,
    behav_name="Wheel Speed"
)
fig.savefig("wheel.png")
fig, ax = plot_behav_raster(
    x_dict=decoding_wheel_data,
    y_dict=mm_wheel_data,
    x_name="Unimodal",
    y_name="Multimodal",
    behav_name="Wheel Speed",
    num_trials=21,
    text_size=20,
    show_colorbar=False,
    show_info=True
)
fig.savefig("wheel_raster.png")
print("wheel done")
# load data
mm_data = np.load(mm_data_path, allow_pickle=True).item()
encoding_data = np.load(encoding_data_path, allow_pickle=True).item()
mm_result = compute_neuron_metrics(data_dict=mm_data)
encoding_result = compute_neuron_metrics(data_dict=encoding_data)
num_neuron = len(mm_result["bps"])
print(f"encode: {encoding_result['bps'].mean()}, mm: {mm_result['bps'].mean()}")
print(f"Number of neurons: {num_neuron}")
bps_diff = mm_result["bps"] - encoding_result["bps"]
sorted_idx = np.argsort(bps_diff)[::-1]
# filter out neurons with low bps in mm_result
sorted_idx = sorted_idx[mm_result["bps"][sorted_idx] > 0.5]
top_3_idx = sorted_idx[:3]
print(sorted_idx)
print(f"mm bps: {mm_result['r2'][top_3_idx]}, encoding bps: {encoding_result['r2'][top_3_idx]}")
fig, ax = plot_multi_neuron_psth(
    x_dict=encoding_data,
    y_dict=mm_data,
    x_name="Unimodal",
    y_name="Multimodal",
    neuron_list=[
        117,79,59, 229, 270,224,
        83,222,150, 283, 140, 259 
    ],
    num_columns=6,
    num_rows=2,
    text_size=20,
    num_trials=100
)
fig.savefig("psth.png")
print("psth done")
fig ,ax = plot_multi_neuron_raster(
    x_dict=encoding_data,
    y_dict=mm_data,
    x_name="Unimodal",
    y_name="Multimodal",
    neuron_list=[117,79,59],
    text_size=20,
    num_trials=21
)
fig.savefig("multi_neuron.png")
print("raster done")
exit()
for neuron_idx in sorted_idx:
    fig, ax = plot_neuron_raster(
        x_dict=encoding_data,
        y_dict=mm_data,
        x_name="Unimodal",
        y_name="Multimodal",
        neuron_idx=neuron_idx,
        text_size=20,
        num_trials=1000
    )
    fig.savefig(f"neuron_{neuron_idx}.png")


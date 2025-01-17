import numpy as np
import os
from utils.metric_utils import (
    compute_neuron_metrics
)
from utils.plot_utils import (
    plot_combined_neuron_psth_raster,
    plot_neuron_raster
)
mm_dir = 'results/sesNum-1_ses-d23a4_set-eval_inModal-spike-choice-block-wheel-whisker_outModal-spike-choice-block-wheel-whisker_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_spike'
encoding_dir = 'results/sesNum-1_ses-d23a4_set-eval_inModal-choice-block-wheel-whisker_outModal-spike_mask-embd_mode-temporal_ratio-0.1_taskVar-random/eval_spike'
mm_data_path = os.path.join(mm_dir, 'data.npy')
encoding_data_path = os.path.join(encoding_dir, 'data.npy')
# load data
mm_data = np.load(mm_data_path, allow_pickle=True).item()
encoding_data = np.load(encoding_data_path, allow_pickle=True).item()
mm_result = compute_neuron_metrics(data_dict=mm_data)
encoding_result = compute_neuron_metrics(data_dict=encoding_data)
num_neuron = len(mm_result["bps"])
print(f"encode: {encoding_result['bps'].mean()}, mm: {mm_result['bps'].mean()}")
print(f"Number of neurons: {num_neuron}")
for neuron_idx in range(num_neuron):
    fig, ax = plot_neuron_raster(
        x_dict=mm_data,
        y_dict=encoding_data,
        x_name="Multimodal",
        y_name="Encoding",
        neuron_idx=neuron_idx,
    )
    fig.savefig(f"neuron_{neuron_idx}.png")
    exit()
    fig = plot_combined_neuron_psth_raster(
        x_dict=mm_data,
        y_dict=encoding_data,
        x_name="Multimodal",
        y_name="Encoding",
        neuron_idx=neuron_idx,
    )
    fig.savefig(f"neuron_{neuron_idx}.png")


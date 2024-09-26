import numpy as np
import argparse
from utils.utils import set_seed, get_npy_files, return_behav_r2, return_spike_bps

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_dir', type=str, default='results')
argparser.add_argument('--model_mode', type=str, default='mm')
argparser.add_argument('--num_sessions', type=int, default=10)
argparser.add_argument('--use_contrastive', action='store_true')
argparser.add_argument('--mixed_training', action='store_true')
argparser.add_argument('--seed', type=int, default=42)

args = argparser.parse_args()
set_seed(42)
avail_behav = ['wheel-speed', 'whisker-motion-energy']

npy_files = get_npy_files(log_dir=args.log_dir, 
                          model_mode=args.model_mode, 
                          num_sessions=args.num_sessions,
                          use_contrastive=args.use_contrastive,
                          mixed_training=True if args.model_mode == 'mm' else False)

behav_result = return_behav_r2(npy_files, avail_behav)
# calculate the mean r2 for each behavior
for behav in avail_behav:
    res = np.array(behav_result[behav])
    print(f"{behav} r2: {np.nanmean(res)}")
discrete_behav = ['choice_acc', 'block_acc']
print(behav_result.keys())
for behav in discrete_behav:
    res = np.array(behav_result[behav])
    print(f"{behav} accuracy: {np.nanmean(res)}")

spike_result = return_spike_bps(npy_files)
print(f"spike bps: {np.nanmean(spike_result)}")
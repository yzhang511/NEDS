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

behav_result, all_behav_dict = return_behav_r2(npy_files, avail_behav)

spike_result, all_spike_dict = return_spike_bps(npy_files)
print('------------------------------------')
with open('data/test_eids.txt') as file:
    test_eids = file.read().splitlines()

for test_eid in test_eids:
    eid = test_eid[:5]
    if eid in all_spike_dict:
        print(f"{eid}: {round(all_spike_dict[eid], 5)}")
print(f"Mean spike bps: {round(np.nanmean(list(all_spike_dict.values())), 5)}")
print()
print(behav_result.keys())
for test_eid in test_eids:
    eid = test_eid[:5]
    if eid in all_behav_dict:
        result_str = ""
        ses_data = list(all_behav_dict[eid])
        for idx in range(len(ses_data)):
            result_str += f"{ses_data[idx]} / "
        print(f"{eid}: {result_str[:-2]}")

behav_res_str = ""
discrete_behav = ['choice_acc', 'block_acc']
for behav in discrete_behav:
    res = np.array(behav_result[behav])
    # print(f"Mean {behav} accuracy: {round(np.nanmean(res), 5)}")
    behav_res_str += f"{round(np.nanmean(res), 5)} / "
# calculate the mean r2 for each behavior
for behav in avail_behav:
    res = np.array(behav_result[behav])
    # print(f"Mean {behav} r2: {round(np.nanmean(res), 5)}")
    behav_res_str += f"{round(np.nanmean(res), 5)} / "
print(f"Mean Behav Results: {behav_res_str[:-2]}")

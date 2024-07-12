import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

result_dir = "/scratch/yl6624/multi_modal_foundation_model/results/ses-db4df448-e449-4a6f-a0e7-288711e7a75a/set-eval/inModal-ap-behavior/outModal-ap-behavior/mask-embd/mode-temporal"
behav_mod = ["wheel-speed", "whisker-motion-energy"]
decoding_res = {}
encoding_res = {}
for mask_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    decoding_path = os.path.join(result_dir, f"ratio-{mask_ratio}", "mixedTraining-True", "modal_behavior", "r2.npy")
    decoding_data = np.load(decoding_path, allow_pickle=True)
    decoding_res[mask_ratio] = decoding_data.item()
    encoding_path = os.path.join(result_dir, f"ratio-{mask_ratio}", "mixedTraining-True","modal_spike", "bps.npy")
    encoding_data = np.load(encoding_path, allow_pickle=True)
    encoding_res[mask_ratio] = np.nanmean(encoding_data)



# plot decoding results in different mask ratios
# plot line chart

linear_encoding_baseline = [
    -1.45
]
linear_decoding_baseline = {
    "wheel-speed": 0.46,
    "whisker-motion-energy": 0.55
}

fourM_encoding_baseline = [
    0.19
]
fourM_decoding_baseline = {
    "wheel-speed": 0.64,
    "whisker-motion-energy": 0.70
}
behav_r2_res = {}
for mask_ratio in decoding_res.keys():
    decoding_data = decoding_res[mask_ratio]
    behav_r2_res[mask_ratio] = {}
    for behav in behav_mod:
        behav_r2_res[mask_ratio][behav] = {}
        for key in decoding_data.keys():
            if "r2_trial" in key and behav in key:
                r2_key = key
            elif "psth" in key and behav in key:
                psth_key = key

        behav_r2_res[mask_ratio][behav]["r2"] = decoding_data[r2_key]
        behav_r2_res[mask_ratio][behav]["psth"] = decoding_data[psth_key]

spike_bps_res = {}
for mask_ratio in encoding_res.keys():
    spike_bps_res[mask_ratio] = encoding_res[mask_ratio]

# plot decoding results in different mask ratios
# plot line chart
color = sns.color_palette("hsv", len(behav_mod))

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for i, behav in enumerate(behav_mod):
    behav_r2 = []
    for mask_ratio in behav_r2_res.keys():
        behav_r2.append(behav_r2_res[mask_ratio][behav]["r2"])
    behav_r2 = np.array(behav_r2)
    ax[0].plot(list(behav_r2_res.keys()), behav_r2, label=behav, color=color[i],)
    # ax[0].axhline(y=linear_decoding_baseline[behav], color='r', linestyle='--', label=f"{behav} Linear Decoding Baseline")
    ax[0].axhline(y=fourM_decoding_baseline[behav], color=color[i], linestyle='--', label=f"{behav} 4M Decoding Baseline")
ax[0].set_xlabel("Mask Ratio")
ax[0].set_ylabel("R2")
ax[0].set_title("Decoding R2")
ax[0].legend()
ax[0].grid()

# plot encoding results in different mask ratios
# plot line chart
ax[1].plot(list(spike_bps_res.keys()), list(spike_bps_res.values()),color='g',)
# ax[1].axhline(y=linear_encoding_baseline, color='r', linestyle='--', label="Linear Encoding Baseline")
ax[1].axhline(y=fourM_encoding_baseline, color='g', linestyle='--', label="4M Encoding Baseline")
ax[1].set_xlabel("Mask Ratio")
ax[1].set_ylabel("BPS")
ax[1].set_title("Encoding BPS")

fig.suptitle("Mask Ratio vs. Decoding R2 and Encoding BPS")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("mask_ratio_vs_decoding_r2_encoding_bps.png")

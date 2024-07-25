import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--result_dir", type=str, default="results")

args = argparser.parse_args()

results_dict = {}

# get all files ending with .npy in the result directory
for root, dirs, files in os.walk(args.result_dir):
    for file in files:
        if file.endswith(".npy"):
            file_path = os.path.join(root, file)
            results_dict[file_path] = np.load(file_path, allow_pickle=True)

# clean up the results_dict
# dict becoms {sesNum-: {ses-: {inModal-: {outModal-: {bps: np.array, r2: np.array}}}}}
cleaned_results_dict = {}
avail_mod = ['ap', 'behavior']
for file_path, data in results_dict.items():
    base_name = os.path.basename(file_path)
    file_path = file_path.replace(args.result_dir, "")
    ses_num = file_path.split("sesNum-")[1].split("/")[0]
    eid = file_path.split("ses-")[1].split("/")[0]
    in_modal = file_path.split("inModal-")[1].split("/")[0]
    out_modal = file_path.split("outModal-")[1].split("/")[0]
    target_modal = file_path.split("modal_")[1].split("/")[0]
    
    if "-" in in_modal and "-" in out_modal:
        model_type = "multi-modal"
    elif "ap" in in_modal and "behavior" in out_modal:
        model_type = "decoding"
    elif "behavior" in in_modal and "ap" in out_modal:
        model_type = "encoding"
    else:
        model_type = "unknown"
    print(in_modal, out_modal, target_modal, model_type)
    cleaned_results_dict[ses_num] = cleaned_results_dict.get(ses_num, {})
    cleaned_results_dict[ses_num][eid] = cleaned_results_dict[ses_num].get(eid, {})
    cleaned_results_dict[ses_num][eid][model_type] = cleaned_results_dict[ses_num][eid].get(model_type, {})
    cleaned_results_dict[ses_num][eid][model_type][target_modal] = cleaned_results_dict[ses_num][eid][model_type].get(target_modal, {})
    
    if target_modal == "behavior":
        if "bps" in base_name:
            continue
        elif "r2" in base_name:
            data = data.item()
            cleaned_results_dict[ses_num][eid][model_type][target_modal] = data
    elif target_modal == "spike":
        if "r2" in base_name:
            r2_psth = np.nanmean(data.T[0])
            r2_per_trial = np.nanmean(data.T[1])
            cleaned_results_dict[ses_num][eid][model_type][target_modal]["spike_r2_psth"] = r2_psth
            cleaned_results_dict[ses_num][eid][model_type][target_modal]["spike_r2_trial"] = r2_per_trial
        elif "bps" in base_name:
            bps = np.nanmean(data).item()
            cleaned_results_dict[ses_num][eid][model_type][target_modal]["bps"] = bps
        else:
            raise ValueError("Unknown file name")
    else:
        raise ValueError("Unknown target modal")

# convert the cleaned_results_dict to a pandas dataframe

df = pd.DataFrame(columns=["sesNum", "ses", "model_type", "target_modal", "bps", "spike_r2_psth", "spike_r2_trial", "wheel-speed_r2_trial", "whisker-motion-energy_r2_trial", "wheel-speed_r2_psth", "whisker-motion-energy_r2_psth"])

for ses_num, ses_data in cleaned_results_dict.items():
    for ses, model_data in ses_data.items():
        for model_type, target_data in model_data.items():
            for target_modal, target_modal_data in target_data.items():
                if target_modal == "behavior":
                    wheel_speed_r2 = target_modal_data["wheel-speed_r2_trial"]
                    whisker_motion_energy_r2 = target_modal_data["whisker-motion-energy_r2_trial"]

                    wheel_speed_r2_psth = target_modal_data["wheel-speed_r2_psth"]
                    whisker_motion_energy_r2_psth = target_modal_data["whisker-motion-energy_r2_psth"]
                    df = df._append({
                        "sesNum": ses_num,
                        "ses": ses,
                        "model_type": model_type,
                        "target_modal": target_modal,
                        "wheel-speed_r2_trial": wheel_speed_r2,
                        "whisker-motion-energy_r2_trial": whisker_motion_energy_r2,
                        "wheel-speed_r2_psth": wheel_speed_r2_psth,
                        "whisker-motion-energy_r2_psth": whisker_motion_energy_r2_psth
                    }, ignore_index=True)
                elif target_modal == "spike":
                    bps = target_modal_data["bps"]
                    spike_r2_psth = target_modal_data["spike_r2_psth"]
                    spike_r2_trial = target_modal_data["spike_r2_trial"]
                    df = df._append({
                        "sesNum": ses_num,
                        "ses": ses,
                        "model_type": model_type,
                        "target_modal": target_modal,
                        "bps": bps,
                        "spike_r2_psth": spike_r2_psth,
                        "spike_r2_trial": spike_r2_trial
                    }, ignore_index=True)
                else:
                    raise ValueError("Unknown target modal")


# create result for each ses
# get unique ses
ses = df["ses"].unique()

for s in ses:
    ses_df = df[df["ses"] == s]
    print(ses_df)
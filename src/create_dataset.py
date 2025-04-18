import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from datasets import (
    load_dataset, load_from_disk, concatenate_datasets
)

from utils.utils import set_seed
from utils.dataset_utils import load_ibl_dataset
from utils.config_utils import config_from_kwargs, update_config

from loader.make_loader import make_loader

logging.basicConfig(level=logging.INFO) 

neural_acronyms = {
    "ap": "spike",
}
static_acronyms = {
    "choice": "choice", 
    "block": "block",
}
dynamic_acronyms = {
    "wheel-speed": "wheel", 
    "whisker-motion-energy": "whisker",
}

model_config = "src/configs/multi_modal/mm.yaml"
kwargs = {"model": f"include:{model_config}"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/multi_modal/trainer_mm.yaml", config)
set_seed(config.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"


# ------ 
# SET UP
# ------ 
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--model_mode", type=str, default="mm")
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mixed_training", action="store_true")
ap.add_argument(
    "--modality", nargs="+", 
    default=["ap", "wheel-speed", "whisker-motion-energy", "choice", "block"]
)
args = ap.parse_args()


eid = args.eid
base_path = args.base_path
model_mode = args.model_mode
modality = args.modality
config["model"]["masker"]["mode"] = args.mask_mode
config["model"]["masker"]["ratio"] = args.mask_ratio

logging.info(f"EID: {eid} model mode: {args.model_mode} mask ratio: {args.mask_ratio}")
logging.info(f"Available modality: {modality}")

neural_mods, static_mods, dynamic_mods = [], [], []
for mod in modality:
    if mod in neural_acronyms:
        neural_mods.append(neural_acronyms[mod])
    elif mod in static_acronyms:
        static_mods.append(static_acronyms[mod])   
    elif mod in dynamic_acronyms:
        dynamic_mods.append(dynamic_acronyms[mod])   

if model_mode == "mm":
    input_mods = output_mods = neural_mods + static_mods + dynamic_mods
elif model_mode == "decoding":
    input_mods = neural_mods
    output_mods = static_mods + dynamic_mods
elif model_mode == "encoding":
    input_mods = static_mods + dynamic_mods
    output_mods = neural_mods
else:
    raise ValueError(f"Model mode {model_mode} not supported.")

modal_filter = {"input": input_mods, "output": output_mods}


# ---------
# LOAD DATA
# ---------
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
    args.data_path,  
    config.dirs.huggingface_org,
    num_sessions=args.num_sessions,
    eid = eid if args.num_sessions == 1 else None,
    use_re=True,
    split_method="predefined",
    test_session_eid=[],
    batch_size=1,
    seed=config.seed
)

max_space_length = max(list(meta_data["eid_list"].values()))
logging.info(f"MAX space length to pad spike data to: {max_space_length}")

train_dataloader = make_loader(
    train_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
    load_meta=config.data.load_meta,
    batch_size=1, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=True
)

base_path = args.data_path
dataset_name = 'ibl_mm' if args.num_sessions == 1 else f'ibl_mm_{args.num_sessions}'

train_save_dir = os.path.join(base_path, dataset_name, 'train')
val_save_dir = os.path.join(base_path, dataset_name, 'val')
test_save_dir = os.path.join(base_path, dataset_name, 'test')
os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(val_save_dir, exist_ok=True)
os.makedirs(test_save_dir, exist_ok=True)

count = 0
for train_data in tqdm(train_dataloader):
    new_dict = {}
    for key, value in train_data.items():
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
        if key == 'neuron_regions':
            neuron_region=[]
            for i in range(len(value)):
                neuron_region.append(value[i][0])
            new_dict[key] = neuron_region
        else:
            new_dict[key] = value[0]
    eid = new_dict['eid']
    save_path = os.path.join(train_save_dir, f"{eid}_{count}")
    # save to npy file
    np.save(save_path, new_dict)
    count += 1

count = 0
val_dataloader = make_loader(
    val_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
    load_meta=config.data.load_meta,
    batch_size=1, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=False
)
for val_data in tqdm(val_dataloader):
    new_dict = {}
    for key, value in val_data.items():
        if type(value) == torch.Tensor:
            # change to numpy array
            value = value.cpu().numpy()
        if key == 'neuron_regions':
            neuron_region=[]
            for i in range(len(value)):
                neuron_region.append(value[i][0])
            new_dict[key] = neuron_region
        else:
            new_dict[key] = value[0]
    eid = new_dict['eid']
    save_path = os.path.join(val_save_dir, f"{eid}_{count}")
    # save to npy file
    np.save(save_path, new_dict)
    count += 1

count = 0
test_dataloader = make_loader(
    test_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
    load_meta=config.data.load_meta,
    batch_size=1, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=False
)
for test_data in tqdm(test_dataloader):
    new_dict = {}
    for key, value in test_data.items():
        if type(value) == torch.Tensor:
            # change to numpy array
            value = value.cpu().numpy()
        if key == 'neuron_regions':
            neuron_region=[]
            for i in range(len(value)):
                neuron_region.append(value[i][0])
            new_dict[key] = neuron_region
        else:
            new_dict[key] = value[0]
    eid = new_dict['eid']
    save_path = os.path.join(test_save_dir, f"{eid}_{count}")
    # save to npy file
    np.save(save_path, new_dict)
    count += 1

logging.info("Finished saving data")

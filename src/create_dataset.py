import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import wandb
import pickle
import logging
import argparse
import threading
import numpy as np
from math import ceil
import torch
from datasets import (
    load_dataset, 
    load_from_disk, 
    concatenate_datasets, 
    load_dataset_builder
)
from utils.dataset_utils import (
    get_user_datasets, 
    load_ibl_dataset, 
    split_both_dataset
)
from datasets import (
    load_dataset, 
    load_from_disk, 
    concatenate_datasets
)
from collections import OrderedDict
from accelerate import Accelerator
from collections import defaultdict
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch_optimizer import Lamb
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from tqdm import tqdm
from accelerate.utils import DistributedDataParallelKwargs

logging.basicConfig(level=logging.INFO) 

neural_acronyms = {
    "ap": "spike",
    "lfp": "lfp",
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
ap.add_argument("--continue_pretrain", action="store_true")
ap.add_argument("--multi_gpu", action="store_true")
ap.add_argument("--debug", action="store_true")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--dummy_load", action="store_true")
ap.add_argument("--dummy_size", type=int, default=50000)
args = ap.parse_args()

if args.debug:
    # Debug using deterministic mode
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logging.info("Deterministic mode is activated. This will negatively impact performance.")

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
import time
start = time.time()
train_save_dir = os.path.join(base_path, dataset_name, 'train')
val_save_dir = os.path.join(base_path, dataset_name, 'val')
test_save_dir = os.path.join(base_path, dataset_name, 'test')
os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(val_save_dir, exist_ok=True)
os.makedirs(test_save_dir, exist_ok=True)
count = 0
for train_data in tqdm(train_dataloader):
    end = time.time()
    # logging.info(f"Time to load one batch: {end-start}")
    start = time.time()
    new_dict = {}
    for key, value in train_data.items():
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
    # for key, value in new_dict.items():
    #     if type(value) == torch.Tensor:
    #         logging.info(f"Key: {key} Shape: {value.shape}")
    #     elif type(value) == list:
    #         logging.info(f"Key: {key} Length: {len(value)}")
    #     elif type(value) == np.ndarray:
    #         logging.info(f"Key: {key} Shape: {value.shape}")
    #     else:
    #         logging.info(f"Key: {key} Type: {type(value)}, Value: {value}")
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

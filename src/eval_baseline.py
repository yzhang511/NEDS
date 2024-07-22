import os
import pickle
import argparse
from math import ceil
import numpy as np
import torch
import wandb
import warnings

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.decoder_embeddings import DecoderEmbedding

from utils.eval_baseline_utils import load_model_data_local, co_smoothing_eval


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='db4df448-e449-4a6f-a0e7-288711e7a75a')
ap.add_argument("--model", type=str, default='linear')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--save_plot", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--wandb', action='store_true')
args = ap.parse_args()

base_path = args.base_path

eid = args.eid

avail_beh = ['wheel-speed', 'whisker-motion-energy']
    
print(f'Working on EID: {eid} ...')

model_config = f"src/configs/baseline.yaml"
n_time_steps = 100
avail_mod = ['ap','behavior']
modal_filter = {
    "input": ['ap'], # 'ap', 'behavior'
    "output": ['behavior'],
    # "input": ['behavior'], # 'ap', 'behavior'
    # "output": ['ap']
}

set_seed(args.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

modal_spike = True if 'ap' in modal_filter['output'] else False
modal_behavior = True if 'behavior' in modal_filter['output'] else False

print('Start model evaluation.')
print('=======================')

model_path = os.path.join(base_path, 
                        "results",
                        f"ses-{eid}",
                        "set-train",
                        f"inModal-{'-'.join(modal_filter['input'])}",
                        f"outModal-{'-'.join(modal_filter['output'])}",
                        args.model,
                        best_ckpt_path
                        )

save_path = os.path.join(base_path,
                        "results",
                        f"ses-{eid}",
                        "set-eval",
                        f"inModal-{'-'.join(modal_filter['input'])}",
                        f"outModal-{'-'.join(modal_filter['output'])}",
                        args.model,
                        )

if args.wandb:
    wandb.init(
        project="multi_modal",
        config=args,
        name="ses-{}_set-eval_inModal-{}_outModal{}-model-{}".format(
            eid[:5], 
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            args.model,
    )
)

configs = {
    'model_config': model_config,
    'model_path': model_path,
    'trainer_config': f'src/configs/trainer.yaml',
    'dataset_path': None, 
    'seed': 42,
    'eid': eid,
    'avail_mod': avail_mod,
    'avail_beh': avail_beh,
}  

model, accelerator, dataset, dataloader = load_model_data_local(**configs)

if modal_spike:
    modal_spike_bps_file = f'{save_path}/modal_spike/bps.npy'
    modal_spike_r2_file = f'{save_path}/modal_spike/r2.npy'
    if not os.path.exists(modal_spike_bps_file) or not os.path.exists(modal_spike_r2_file) or args.overwrite:
        print('Start modal_spike:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'save_path': f'{save_path}/modal_spike',
            'mode': 'modal_spike',
            'n_time_steps': n_time_steps,  
            'held_out_list': list(range(0, 100)),
            'is_aligned': True,
            'target_regions': None,
            'modal_filter': modal_filter,
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping modal_spike since files exist or overwrite is False")

if modal_behavior:
    modal_behavior_bps_file = f'{save_path}/modal_behavior/bps.npy'
    modal_behavior_r2_file = f'{save_path}/modal_behavior/r2.npy'
    if not os.path.exists(modal_behavior_bps_file) or not os.path.exists(modal_behavior_r2_file) or args.overwrite:
        print('Start modal_behavior:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'save_path': f'{save_path}/modal_behavior',
            'mode': 'modal_behavior',
            'n_time_steps': n_time_steps,  
            'held_out_list': list(range(0, 100)),
            'is_aligned': True,
            'target_regions': None,
            'avail_beh': avail_beh,
            'modal_filter': modal_filter,
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping modal_behavior since files exist or overwrite is False")

print('Finish model evaluation.')
print('=======================')

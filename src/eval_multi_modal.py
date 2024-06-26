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

from utils.eval_utils import load_model_data_local, spiking_activity_recon_eval, behavior_recon_eval, co_smoothing_eval


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='671c7ea7-6726-4fbe-adeb-f89c2c8e489b')
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--use_MtM", action='store_true')
ap.add_argument("--cont_target", type=str, default="whisker-motion-energy")
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--save_plot", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--wandb', action='store_true')
args = ap.parse_args()

base_path = args.base_path

eid = args.eid
    
print(f'Working on EID: {eid} ...')

model_config = f"src/configs/multi_modal/mm.yaml"
mask_name = f"mask_{args.mask_mode}"
n_time_steps = 100
avail_mod = ['ap']

if args.wandb:
    wandb.init(
        project="multi_modal",
        config=args,
        name="{}_eval_multi_modal_mask_{}_ratio_{}_useMtM_{}".format(
            eid[:5], args.mask_mode, args.mask_ratio, args.use_MtM
        )
    )

set_seed(args.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

spike_recon = False
behave_recon = True if 'behavior' in avail_mod else False
co_smooth = True if 'ap' in avail_mod else False
forward_pred = True if 'ap' in avail_mod else False
inter_region = False
intra_region = False

print('Start model evaluation.')
print('=======================')

model_path = f'{base_path}/results/{eid}/train/multi_modal/mask_{args.mask_mode}/ratio_{args.mask_ratio}/useMtM_{args.use_MtM}/{best_ckpt_path}'
save_path = f'{base_path}/results/{eid}/eval/multi_modal/mask_{args.mask_mode}/ratio_{args.mask_ratio}/useMtM_{args.use_MtM}/'

configs = {
    'model_config': model_config,
    'model_path': model_path,
    'trainer_config': f'src/configs/multi_modal/trainer_mm.yaml',
    'dataset_path': None, 
    'seed': 42,
    'mask_name': mask_name,
    'eid': eid,
    'avail_mod': avail_mod,
    'avail_beh': [args.cont_target],
}  

model, accelerator, dataset, dataloader = load_model_data_local(**configs)

print("(eval) masking ratio: ", model.masker.ratio)
print("(eval) masking active: ", model.masker.force_active)
print("(eval) masking mask regions: ", model.masker.mask_regions)
print("(eval) masking target regions: ", model.masker.target_regions)

if spike_recon:
    spike_recon_bps_file = f'{save_path}/spike_recon/bps.npy'
    spike_recon_r2_file = f'{save_path}/spike_recon/r2.npy'
    if not os.path.exists(spike_recon_bps_file) or not os.path.exists(spike_recon_r2_file) or args.overwrite:
        print('Start spike_recon:')
        spike_recon_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/spike_recon',
            'n_time_steps': n_time_steps,    
            'is_aligned': True,
            'target_regions': None,
        }
        results = spiking_activity_recon_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **spike_recon_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping spike_recon since files exist or overwrite is False")


if behave_recon:
    behave_recon_bps_file = f'{save_path}/behave_recon/bps.npy'
    behave_recon_r2_file = f'{save_path}/behave_recon/r2.npy'
    if not os.path.exists(behave_recon_bps_file) or not os.path.exists(behave_recon_r2_file) or args.overwrite:
        print('Start behave_recon:')
        behave_recon_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/behave_recon',
            'n_time_steps': n_time_steps,    
            'is_aligned': True,
            'target_regions': None,
            'avail_beh': [args.cont_target],
        }
        results = behavior_recon_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **behave_recon_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping behave_recon since files exist or overwrite is False")



if co_smooth:
    co_smooth_bps_file = f'{save_path}/co_smooth/bps.npy'
    co_smooth_r2_file = f'{save_path}/co_smooth/r2.npy'
    if not os.path.exists(co_smooth_bps_file) or not os.path.exists(co_smooth_r2_file) or args.overwrite:
        print('Start co-smoothing:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/co_smooth',
            'mode': 'per_neuron',
            'n_time_steps': n_time_steps,    
            'is_aligned': True,
            'target_regions': None,
        }
        results = co_smoothing_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping co_smoothing since files exist or overwrite is False")


if forward_pred:
    forward_pred_bps_file = f'{save_path}/forward_pred/bps.npy'
    forward_pred_r2_file = f'{save_path}/forward_pred/r2.npy'
    if not os.path.exists(forward_pred_bps_file) or not os.path.exists(forward_pred_r2_file) or args.overwrite:
        print('Start forward_pred:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/forward_pred',
            'mode': 'forward_pred',
            'n_time_steps': n_time_steps,  
            'held_out_list': list(range(90, 100)),
            'is_aligned': True,
            'target_regions': None,
        }
        results = co_smoothing_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping forward_pred since files exist or overwrite is False")


if inter_region:
    inter_region_bps_file = f'{save_path}/inter_region/bps.npy'
    inter_region_pred_r2_file = f'{save_path}/inter_region/r2.npy'
    if not os.path.exists(inter_region_bps_file) or not os.path.exists(inter_region_r2_file) or args.overwrite:
        print('Start inter_region:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/inter_region',
            'mode': 'inter_region',
            'n_time_steps': n_time_steps,   
            'held_out_list': None,
            'is_aligned': True,
            'target_regions': None,
        }
        results = co_smoothing_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping inter_region since files exist or overwrite is False")


if intra_region:
    intra_region_bps_file = f'{save_path}/intra_region/bps.npy'
    intra_region_pred_r2_file = f'{save_path}/intra_region/r2.npy'
    if not os.path.exists(intra_region_bps_file) or not os.path.exists(intra_region_r2_file) or args.overwrite:
        print('Start intra_region:')
        co_smoothing_configs = {
            'subtract': 'task',
            'onset_alignment': [40],
            'method_name': mask_name, 
            'save_path': f'{save_path}/intra_region',
            'mode': 'intra_region',
            'n_time_steps': n_time_steps,    
            'held_out_list': None,
            'is_aligned': True,
            'target_regions': ['all'],
        }
        results = co_smoothing_eval(
            model, accelerator, 
            dataloader, dataset, 
            save_plot=args.save_plot,
            use_mtm=args.use_MtM,
            **co_smoothing_configs
        )
        print(results)
        wandb.log(results)
    else:
        print("skipping intra_region since files exist or overwrite is False")
        

print('Finish model evaluation.')
print('=======================')


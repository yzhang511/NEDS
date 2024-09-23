import os
import pickle
import argparse
from math import ceil
import numpy as np
import torch
import wandb
import warnings
import threading

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.decoder_embeddings import DecoderEmbedding

from utils.eval_utils import load_model_data_local

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='db4df448-e449-4a6f-a0e7-288711e7a75a')
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--mask_type", type=str, default="input")
ap.add_argument("--use_MtM", action='store_true')
ap.add_argument("--mixed_training", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--dummy_load", action='store_true')
ap.add_argument("--dummy_size", type=int, default=50000)
ap.add_argument("--model_mode", type=str, default="mm")
ap.add_argument("--use_contrastive", action='store_true')

args = ap.parse_args()

base_path = args.base_path

eid = args.eid

avail_beh = ['wheel-speed', 'whisker-motion-energy']
    
print(f'Working on EID: {eid} ...')
if args.use_contrastive:
    model_config = "src/configs/multi_modal/mm_contrastive.yaml"
else:
    model_config = "src/configs/multi_modal/mm.yaml"
kwargs = {
    "model": f"include:{model_config}",
}

config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/multi_modal/trainer_mm.yaml", config)

config['model']['masker']['mode'] = args.mask_mode
config['model']['masker']['ratio'] = args.mask_ratio
set_seed(config.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

avail_mod = ['ap', 'behavior']

if args.model_mode == "mm":
    input_modal = ['ap', 'behavior']
    output_modal = ['ap', 'behavior']
elif args.model_mode == "decoding":
    input_modal = ['ap']
    output_modal = ['behavior']
elif args.model_mode == "encoding":
    input_modal = ['behavior']
    output_modal = ['ap']
else:
    raise ValueError(f"model_mode {args.model_mode} not supported")

modal_filter = {
    "input": input_modal,
    "output": output_modal
}

train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                                    config.dirs.huggingface_org,
                                    num_sessions=1,
                                    eid = args.eid,
                                    use_re=True,
                                    split_method="predefined",
                                    test_session_eid=[],
                                    batch_size=config.training.train_batch_size,
                                    seed=config.seed)

num_sessions = args.num_sessions
mask_mode = '-'.join(config.training.mask_mode) if config.training.mask_type == 'input' else args.mask_mode
mask_name = f"mask_{args.mask_mode}"

log_dir = os.path.join(base_path, 
                       "results",
                       f"sesNum-{num_sessions}",
                       f"ses-{args.eid[:5]}",
                       "set-finetune",
                       f"inModal-{'-'.join(modal_filter['input'])}",
                       f"outModal-{'-'.join(modal_filter['output'])}",
                       f"mask-{config.training.mask_type}",
                       f"mode-{mask_mode}",
                       f"ratio-{args.mask_ratio}",
                       f"mixedTraining-{args.mixed_training}",
                       f"contrast-{config.model.use_contrastive}",
                       )
final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, "last checkpoint exists and overwrite is False"
os.makedirs(log_dir, exist_ok=True)
if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="sesNum-{}_ses-{}_set-finetune_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_mixedTraining-{}_contrastive-{}".format(
            num_sessions,
            args.eid[:5], 
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            config.training.mask_type, 
            mask_mode,
            args.mask_ratio,
            args.mixed_training,
            config.model.use_contrastive
        )
    )

print('Start model training.')
print('=====================')

n_behaviors = len(avail_beh)

train_dataloader = make_loader(train_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.train_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=meta_data['num_neurons'][0],
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            stitching=True,
                            seed=config.seed,
                            shuffle=True)

val_dataloader = make_loader(val_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.test_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=meta_data['num_neurons'][0],
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            stitching=True,
                            seed=config.seed,
                            shuffle=False)

test_dataloader = make_loader(test_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.test_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=meta_data['num_neurons'][0],
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            stitching=True,
                            seed=config.seed,
                            shuffle=False)

accelerator = Accelerator()

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

eid_ = "multi"

if args.model_mode == 'mm':
    avg_state_dict = []
    for best_ckpt_path in ['model_best.pt', 'model_best_spike.pt', 'model_best_behave.pt']:
        model_path = os.path.join(base_path, 
                                "results",
                                f"sesNum-{args.num_sessions}",
                                f"ses-{eid_}",
                                "set-train",
                                f"inModal-{'-'.join(modal_filter['input'])}",
                                f"outModal-{'-'.join(modal_filter['output'])}",
                                f"mask-{args.mask_type}",
                                f"mode-{mask_mode}",
                                f"ratio-{args.mask_ratio}",
                                f"mixedTraining-{args.mixed_training}",
                                f"contrast-{args.use_contrastive}",
                                best_ckpt_path
                                )
        
        configs = {
            'model_config': model_config,
            'model_path': model_path,
            'trainer_config': f'src/configs/multi_modal/trainer_mm.yaml',
            'dataset_path': None, 
            'seed': 42,
            'mask_name': mask_name,
            'eid': args.eid,
            'avail_mod': avail_mod,
            'avail_beh': avail_beh,
        }  
        
        model, accelerator, dataset, dataloader = load_model_data_local(**configs)
        model_state_dict = model.state_dict()
        avg_state_dict.append(model_state_dict)

    for key in model_state_dict:
        model_state_dict[key] = sum([state_dict[key] for state_dict in avg_state_dict]) / len(avg_state_dict)
    model.load_state_dict(model_state_dict)

elif args.model_mode in ['encoding', 'decoding']:
    model_path = os.path.join(base_path, 
                        "results",
                        f"sesNum-{args.num_sessions}",
                        f"ses-{eid_}",
                        "set-train",
                        f"inModal-{'-'.join(modal_filter['input'])}",
                        f"outModal-{'-'.join(modal_filter['output'])}",
                        f"mask-{args.mask_type}",
                        f"mode-{mask_mode}",
                        f"ratio-{args.mask_ratio}",
                        f"mixedTraining-{args.mixed_training}",
                        f"contrast-{args.use_contrastive}",
                        best_ckpt_path
                        )

    configs = {
        'model_config': model_config,
        'model_path': model_path,
        'trainer_config': f'src/configs/multi_modal/trainer_mm.yaml',
        'dataset_path': None, 
        'seed': 42,
        'mask_name': mask_name,
        'eid': args.eid,
        'avail_mod': avail_mod,
        'avail_beh': avail_beh,
    }  
    
    model, accelerator, dataset, dataloader = load_model_data_local(**configs)
model.masker.ratio = args.mask_ratio 
print("(train) masking mode: ", model.masker.mode)
print("(train) masking ratio: ", model.masker.ratio)
print("(train) masking active: ", model.masker.force_active)


# change stitcher
encoder_embeddings, decoder_embeddings = {}, {}

for mod in modal_filter["input"]:
    model.encoder_embeddings[mod] = EncoderEmbedding(
        hidden_size=config.model.encoder.transformer.hidden_size,
        n_channel=256 if mod=='ap' else 256,
        stitching=True,
        eid_list=meta_data['eid_list'],
        mod=mod,
        config=config.model.encoder,
    )

for mod in modal_filter["output"]:
    model.decoder_embeddings[mod] = DecoderEmbedding(
        hidden_size=config.model.decoder.transformer.hidden_size,
        #####
        n_channel=(256+(256//2)) if len(modal_filter['output']) > 1 else 256,
        output_channel=(256+(256//2)) if len(modal_filter['output']) > 1 else 256,
        #####
        stitching=True,
        eid_list=meta_data['eid_list'],
        mod=mod,
        config=config.model.decoder,
    )
    
model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.optimizer.lr, 
    weight_decay=config.optimizer.wd, 
    eps=config.optimizer.eps
)

lr_scheduler = OneCycleLR(
    optimizer=optimizer,
    total_steps=config.training.num_epochs*len(train_dataloader)//config.optimizer.gradient_accumulation_steps,
    max_lr=config.optimizer.lr,
    pct_start=config.optimizer.warmup_pct,
    div_factor=config.optimizer.div_factor,
)

trainer_kwargs = {
    "log_dir": log_dir,
    "accelerator": accelerator,
    "lr_scheduler": lr_scheduler,
    "avail_mod": avail_mod,
    "modal_filter": modal_filter,
    "mixed_training": args.mixed_training,
    "config": config,
}

# Shared variable to signal the dummy load to stop
stop_dummy_load = threading.Event()
trainer_ = make_multimodal_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    **trainer_kwargs,
    **meta_data
)

if args.dummy_load:
    # Start the dummy load
    print(f"Starting dummy load with {args.dummy_size} samples")
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load, args.dummy_size))
    dummy_thread.start()

    try:
        trainer_.train()
    finally:
        stop_dummy_load.set()
        dummy_thread.join()
else:
    trainer_.train()

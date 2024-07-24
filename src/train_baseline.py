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
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_baseline_trainer
from models.baseline_encoder import BaselineEncoder, ReducedRankEncoder
from models.baseline_decoder import BaselineDecoder, ReducedRankDecoder


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='db4df448-e449-4a6f-a0e7-288711e7a75a')
ap.add_argument("--model", type=str, default='linear')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
args = ap.parse_args()

base_path = args.base_path

eid = args.eid

avail_beh = ['wheel-speed', 'whisker-motion-energy']
    
print(f'Working on EID: {eid} ...')

kwargs = {
    "model": f"include:src/configs/baseline.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/trainer.yaml", config)

set_seed(config.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

avail_mod = ['ap', 'behavior']

modal_filter = {
    "input": ['ap'], 
    "output": ['behavior']
}

log_dir = os.path.join(base_path, 
                       "results",
                       f"ses-{eid}",
                       "set-train",
                       f"inModal-{'-'.join(modal_filter['input'])}",
                       f"outModal-{'-'.join(modal_filter['output'])}",
                       args.model,
                       )
final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, "last checkpoint exists and overwrite is False"

if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="ses-{}_set-train_inModal-{}_outModal-{}_model-{}".format(
            eid[:5], 
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            args.model
        )
    )
os.makedirs(log_dir, exist_ok=True)
_, _, _, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                    config.dirs.huggingface_org,
                    eid=eid,
                    num_sessions=1,
                    split_method="predefined",
                    test_session_eid=[],
                    batch_size=config.training.train_batch_size,
                    seed=config.seed)
print(meta_data)

print('Start model training.')
print('=====================')

dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
train_dataset = dataset["train"]
val_dataset = dataset["val"]
test_dataset = dataset["test"]
print(dataset.column_names)

n_behaviors = len(avail_beh)
n_neurons = len(train_dataset['cluster_regions'][0])
meta_data['num_neurons'] = [n_neurons]
print(meta_data)

train_dataloader = make_loader(train_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.train_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=n_neurons,
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            shuffle=True)

val_dataloader = make_loader(val_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.test_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=n_neurons,
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            shuffle=False)

test_dataloader = make_loader(test_dataset, 
                            target=avail_beh,
                            load_meta=config.data.load_meta,
                            batch_size=config.training.test_batch_size, 
                            pad_to_right=True, 
                            pad_value=-1.,
                            max_time_length=config.data.max_time_length,
                            max_space_length=n_neurons,
                            dataset_name=config.data.dataset_name,
                            sort_by_depth=config.data.sort_by_depth,
                            sort_by_region=config.data.sort_by_region,
                            shuffle=False)

encoder_embeddings, decoder_embeddings = {}, {}

if "ap" in modal_filter["output"]:
    if args.model == 'linear':
        model_class = "LinearEncoder" 
    elif args.model == 'rrr':
        model_class = "ReducedRankEncoder" 
        meta_data["rank"] = 4 
    else:
        raise NotImplementedError
    input_size = n_behaviors
    output_size = n_neurons
else: 
    if args.model == 'linear':
        model_class = "LinearDecoder" 
    elif args.model == 'rrr':
        model_class = "ReducedRankDecoder" 
        meta_data["rank"] = 4 
    else:
        raise NotImplementedError
    input_size = n_neurons 
    output_size = n_behaviors

accelerator = Accelerator()

NAME2MODEL = {
    "LinearEncoder": BaselineEncoder, "ReducedRankEncoder": ReducedRankEncoder,
    "LinearDecoder": BaselineDecoder, "ReducedRankDecoder": ReducedRankDecoder,
}
model_class = NAME2MODEL[model_class]
model = model_class(
    in_channel=input_size, 
    out_channel=output_size,
    **config.method.model_kwargs, 
    **meta_data
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
    "config": config,
}

trainer_ = make_baseline_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    **trainer_kwargs,
    **meta_data
)
trainer_.train()

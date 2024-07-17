import os
import pickle
import argparse
from argparse import Namespace
from math import ceil
import numpy as np
import torch
import wandb
import warnings
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from accelerate import Accelerator

from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from utils.dataset_utils import load_ibl_dataset
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.decoder_embeddings import DecoderEmbedding
from models.glvm import ModelGLVM



ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='db4df448-e449-4a6f-a0e7-288711e7a75a')
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--use_MtM", action='store_true')
ap.add_argument("--mixed_training", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
args = ap.parse_args()

base_path = args.base_path

eid = args.eid

avail_beh = ['wheel-speed', 'whisker-motion-energy']
    
print(f'Working on EID: {eid} ...')

kwargs = {
    "model": f"include:src/configs/multi_modal/mm.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config(f"src/configs/multi_modal/trainer_mm.yaml", config)

config['model']['masker']['mode'] = args.mask_mode
config['model']['masker']['ratio'] = args.mask_ratio
set_seed(config.seed)

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'

avail_mod = ['ap', 'behavior']

modal_filter = {
    "input": ['behavior'], 
    "output": ['ap']
}

if config.training.mask_type == 'input':
    mask_mode = '-'.join(config.training.mask_mode)
else:
    mask_mode = args.mask_mode

with open("src/configs/glvm/train_config.yaml", "r") as f:
    data_conf = yaml.load(f, Loader=yaml.Loader)
# update the config file
# the second overwrites the first - comman line will beat config
args_dict = {**data_conf.__dict__}
glvm_args = Namespace(**args_dict)
methods = [
            # "zero_imputation_mask_concatenated_encoder_only",
            "zero_imputation",
        ]
logs = {
            method: {
                "elbo": [],
                "kl": [],
                "rec_loss": [],
                "elbo_val": [],
                "kl_val": [],
                "rec_loss_val": [],
                "masked_rec_loss": [],
                "masked_rec_loss_val": [],
                "observed_mse": [],
                "masked_mse": [],
                "time_stamp_val": [],
            }
            for method in methods
        }
model = ModelGLVM(
                args=glvm_args,
                logs=logs["zero_imputation"],
            )
print(model)
exit()
log_dir = os.path.join(base_path, 
                       "results",
                       f"ses-{eid}",
                       "set-train",
                       f"inModal-{'-'.join(modal_filter['input'])}",
                       f"outModal-{'-'.join(modal_filter['output'])}",
                       f"mask-input",
                       f"mode-gaussian",
                       f"ratio-{args.mask_ratio}",
                       f"mixedTraining-{args.mixed_training}"
                       )

final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, "last checkpoint exists and overwrite is False"

if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_mixedTraining-{}".format(
            eid[:5], 
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            config.training.mask_type, 
            mask_mode,
            args.mask_ratio,
            args.mixed_training
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

for mod in modal_filter["input"]:
    encoder_embeddings[mod] = EncoderEmbedding(
        hidden_size=config.model.encoder.transformer.hidden_size,
        n_channel=n_neurons if mod == 'ap' else n_behaviors,
        config=config.model.encoder,
    )

for mod in modal_filter["output"]:
    decoder_embeddings[mod] = DecoderEmbedding(
        hidden_size=config.model.decoder.transformer.hidden_size,
        n_channel=n_neurons if mod == 'ap' else n_behaviors,
        output_channel=n_neurons if mod == 'ap' else n_behaviors,
        config=config.model.decoder,
    )

accelerator = Accelerator()

print("(train) masking mode: ", model.masker.mode)
print("(train) masking ratio: ", model.masker.ratio)
print("(train) masking active: ", model.masker.force_active)

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

trainer_ = make_multimodal_trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    **trainer_kwargs,
    **meta_data
)
trainer_.train()

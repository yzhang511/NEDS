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
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_baseline_trainer
from models.baseline_encoder import BaselineEncoder, ReducedRankEncoder
from models.baseline_decoder import BaselineDecoder, ReducedRankDecoder


# -----------
# USER INPUTS
# -----------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="db4df448-e449-4a6f-a0e7-288711e7a75a")
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--model_mode", type=str, default="decoding")
args = ap.parse_args()


# ------
# CONFIG
# ------
eid = args.eid
base_path = args.base_path
avail_beh = args.behavior 
avail_mod = args.modality
num_sessions = args.num_sessions
n_behaviors = len(avail_beh)
if num_sessions > 1:
    assert args.model != "linear", "Linear baselines do not support multi-session training."

kwargs = {
    "model": f"include:src/configs/baseline.yaml"
}
config = config_from_kwargs(kwargs)
if args.model_mode == "decoding":
    config = update_config(f"src/configs/trainer_decoder.yaml", config)
elif args.model_mode == "encoding":
    config = update_config(f"src/configs/trainer_encoder.yaml", config)

set_seed(config.seed)

if args.model_mode == "decoding":
    input_modal = ['ap']
    output_modal = ['behavior']
elif args.model_mode == "encoding":
    input_modal = ['behavior']
    output_modal = ['ap']
else:
    raise ValueError(f"model_mode {args.model_mode} not supported")
    
modal_filter = {"input": input_modal, "output": output_modal}


# ---------
# LOAD DATA
# ---------
eid_ = args.eid if args.num_sessions == 1 else None

train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
    config.dirs.dataset_cache_dir, 
    config.dirs.huggingface_org,
    num_sessions=args.num_sessions,
    eid = eid_,
    use_re=True,
    split_method="predefined",
    test_session_eid=[],
    batch_size=config.training.train_batch_size,
    seed=config.seed
)

num_sessions = len(meta_data['eid_list'])
eid_ = "multi" if num_sessions > 1 else eid[:5]

last_ckpt_path = 'model_last.pt'
best_ckpt_path = 'model_best.pt'
log_dir = os.path.join(base_path,
                       "results",
                       f"sesNum-{num_sessions}",
                       f"ses-{eid_}",
                       "set-train",
                       f"inModal-{'-'.join(modal_filter['input'])}",
                       f"outModal-{'-'.join(modal_filter['output'])}",
                       args.model
                       )
final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, "last checkpoint exists and overwrite is False"
os.makedirs(log_dir, exist_ok=True)

if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_model-{}".format(
            num_sessions,
            eid_,
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            args.model
        )
    )

train_dataloader = make_loader(
    train_dataset, 
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
    shuffle=True
)

val_dataloader = make_loader(
    val_dataset, 
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
    shuffle=False
)

test_dataloader = make_loader(
    test_dataset, 
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
    shuffle=False
)

# --------
# TRAINING
# --------
encoder_embeddings, decoder_embeddings = {}, {}

if "ap" in modal_filter["output"]:
    if args.model == 'linear':
        model_class = "LinearEncoder" 
    elif args.model == 'rrr':
        model_class = "ReducedRankEncoder" 
        meta_data["rank"] = 4 
    else:
        raise NotImplementedError
    input_size, output_size = n_behaviors, meta_data['num_neurons']
else: 
    if args.model == 'linear':
        model_class = "LinearDecoder" 
    elif args.model == 'rrr':
        model_class = "ReducedRankDecoder" 
        meta_data["rank"] = 4 
    else:
        raise NotImplementedError
    input_size, output_size = meta_data['num_neurons'], n_behaviors

accelerator = Accelerator()

NAME2MODEL = {
    "LinearEncoder": BaselineEncoder, "ReducedRankEncoder": ReducedRankEncoder,
    "LinearDecoder": BaselineDecoder, "ReducedRankDecoder": ReducedRankDecoder,
}
model_class = NAME2MODEL[model_class]

print(meta_data)
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

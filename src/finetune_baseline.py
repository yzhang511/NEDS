import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import wandb
import pickle
import logging
import argparse
import numpy as np
from math import ceil
import torch
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from trainer.make import make_baseline_trainer
from models.baseline_encoder import BaselineEncoder, ReducedRankEncoder
from models.baseline_decoder import BaselineDecoder, ReducedRankDecoder

from torch.cuda.amp import GradScaler

logging.basicConfig(level=logging.INFO) 

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel-speed", "whisker-motion-energy"]

model_config = "src/configs/baseline.yaml"
if args.model_mode == "decoding":
    trainer_config = update_config("src/configs/trainer_decoder.yaml", model_config)
elif args.model_mode == "encoding":
    trainer_config = update_config("src/configs/trainer_encoder.yaml", model_config)
set_seed(trainer_config.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

# ------ 
# SET UP
# ------ 

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--model_mode", type=str, default="decoding")
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--rank", type=int, default=4)
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()


eid = args.eid
base_path = args.base_path
avail_beh = args.behavior 
avail_mod = args.modality
model_mode = args.model_mode
model_class = args.model
n_beh = len(avail_beh)

assert model_class == "rrr", "Finetuning only supported for reduced rank model."

if args.model_mode == "decoding":
    input_modal = ["ap"]
    output_modal = ["behavior"]
elif args.model_mode == "encoding":
    input_modal = ["behavior"]
    output_modal = ["ap"]
else:
    raise ValueError(f"Model mode {model_mode} not supported.")
    
modal_filter = {"input": input_modal, "output": output_modal}


# ---------
# LOAD DATA
# ---------
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
    config.dirs.dataset_cache_dir, 
    config.dirs.huggingface_org,
    num_sessions=1,
    eid=eid,
    use_re=True,
    split_method="predefined",
    test_session_eid=[],
    batch_size=config.training.train_batch_size,
    seed=config.seed
)

train_dataloader = make_loader(
    train_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.train_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=meta_data["num_neurons"][0],
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=True
)

val_dataloader = make_loader(
    val_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=meta_data["num_neurons"][0],
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=False
)

test_dataloader = make_loader(
    test_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=meta_data["num_neurons"][0],
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    shuffle=False
)

# --------
# SET PATH
# --------
num_sessions = args.num_sessions

pretrain_path = \
"sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_model-{}".format(
    num_sessions,
    "multi",
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    f"behavior-{'-'.join(avail_beh)}",
    model_class,
)

log_name = "sesNum-{}_ses-{}_set-finetune_inModal-{}_outModal-{}_model-{}".format(
    num_sessions,
    eid[:5],
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    f"behavior-{'-'.join(avail_beh)}",
    model_class,
)

log_dir = os.path.join(base_path, "results", log_name)

logging.info(f"Save model to {log_dir}")

final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, \
    "last checkpoint exists and overwrite is False"
os.makedirs(log_dir, exist_ok=True)

if config.wandb.use:
    wandb.init(
        project=config.wandb.project, 
        entity=config.wandb.entity, 
        config=config,
        name=log_name
    )


# ----------
# LOAD MODEL
# ----------
logging.info(f"Start model finetuning:")

accelerator = Accelerator()

model_path = os.path.join(
    base_path, "results", pretrain_path, best_ckpt_path
)   
configs = {
    "model_config": model_config,
    "model_path": pretrain_path,
    "trainer_config": trainer_config,
    "dataset_path": None, 
    "seed": trainer_config.seed,
    "eid": eid,
    "avail_mod": avail_mod,
    "avail_beh": avail_beh,
}  
model, accelerator, dataset, dataloader = load_model_data_local(**configs)
model_state_dict = model.state_dict()
model.load_state_dict(model_state_dict)

# -----------------------
# ACCOMMODATE NEW SESSION
# -----------------------

Us, bs = {}, {}
if model_mode == "decoding":
    for key, val in meta_data["eid_list"].items():
        Us[str(key)] = torch.nn.Parameter(torch.randn(val, model.rank))
        bs[str(key)] = torch.nn.Parameter(torch.randn(model.out_channel,))
elif model_mode == "encoding":
    for key, val in meta_data["eid_list"].items():
        Us[str(key)] = torch.nn.Parameter(torch.randn(val, model.in_channel, model.rank))
        bs[str(key)] = torch.nn.Parameter(torch.randn(val,))
else:
    raise NotImplementedError
model.Us = torch.nn.ParameterDict(Us)
model.bs = torch.nn.ParameterDict(bs)

# ------------
# SET UP MODEL
# ------------
model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.optimizer.lr, 
    weight_decay=config.optimizer.wd, 
    eps=config.optimizer.eps
)

grad_accum_steps = config.optimizer.gradient_accumulation_steps

lr_scheduler = OneCycleLR(
    optimizer=optimizer,
    total_steps=config.training.num_epochs*len(train_dataloader)//grad_accum_steps,
    max_lr=config.optimizer.lr,
    pct_start=config.optimizer.warmup_pct,
    div_factor=config.optimizer.div_factor,
)

scaler = GradScaler()

# -----
# TRAIN
# -----
trainer_kwargs = {
    "log_dir": log_dir,
    "accelerator": accelerator,
    "lr_scheduler": lr_scheduler,
    "scaler": scaler,
    "avail_mod": avail_mod,
    "modal_filter": modal_filter,
    "config": config,
    "target_to_decode": avail_beh,
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

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
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(level=logging.INFO) 


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--model_mode", type=str, default="decoding")
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--rank", type=int, default=4)
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--use_nlb", action="store_true")
args = ap.parse_args()

kwargs = {"model": "include:src/configs/baseline.yaml"}
config = config_from_kwargs(kwargs)
if args.model_mode == "decoding":
    config = update_config("src/configs/trainer_decoder.yaml", config)
elif args.model_mode == "encoding":
    config = update_config("src/configs/trainer_encoder.yaml", config)
set_seed(config.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

if not args.use_nlb:
    avail_beh = args.behavior
    STATIC_VARS = ["choice", "block"]
    DYNAMIC_VARS = ["wheel-speed", "whisker-motion-energy"]
else:
    avail_beh = ["finger_vel"]
    STATIC_VARS = []
    DYNAMIC_VARS = ["finger_vel"]

OUTPUT_DIM = {
    "choice": 2, "block": 3, "wheel": 1, "whisker": 1, 
    "cursor_pos": 2, "target_pos": 2, "finger_pos": 3, "finger_vel": 2
}

# ------ 
# SET UP
# ------
eid = args.eid
base_path = args.base_path
avail_beh = args.behavior if not args.use_nlb else avail_beh
avail_mod = args.modality if not args.use_nlb else ["spike", "behavior"]
model_mode = args.model_mode
model_class = args.model
n_beh = len(avail_beh)

if args.model_mode == "decoding":
    input_modal = ["ap"] if not args.use_nlb else ["spike"]
    output_modal = ["behavior"]
elif args.model_mode == "encoding":
    input_modal = ["behavior"]
    output_modal = ["ap"] if not args.use_nlb else ["spike"]
else:
    raise ValueError(f"Model mode {model_mode} not supported.")

modal_filter = {"input": input_modal, "output": output_modal}


# ---------
# LOAD DATA
# ---------
if not args.use_nlb:
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
        config.dirs.dataset_cache_dir, 
        config.dirs.huggingface_org,
        num_sessions=args.num_sessions,
        eid = eid if args.num_sessions == 1 else None,
        use_re=True,
        split_method="predefined",
        test_session_eid=[],
        batch_size=config.training.train_batch_size,
        seed=config.seed
    )
else:
    from utils.nlb_data_utils import load_nlb_dataset
    trial_length, bin_size = 600, 20 # ms
    train_dataset, val_dataset, test_dataset, meta_data = load_nlb_dataset(
        "/projects/beez/yzhang39/nlb/000129/sub-Indy", 20
    )
    config["data"]["max_time_length"] = int(trial_length / bin_size)

max_space_length = max(list(meta_data["eid_list"].values()))
logging.info(f"MAX space length to pad spike data to: {max_space_length}")

local_data_dir = "ibl_mm" if args.num_sessions == 1 else f"ibl_mm_{args.num_sessions}"

train_dataloader = make_loader(
    train_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.train_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="train",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    shuffle=True,
    use_nlb=args.use_nlb
)

val_dataloader = make_loader(
    val_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="val",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    shuffle=False,
    use_nlb=args.use_nlb
)

test_dataloader = make_loader(
    test_dataset, 
    target=DYNAMIC_VARS,
    load_meta=config.data.load_meta,
    batch_size=config.training.test_batch_size, 
    pad_to_right=True, 
    pad_value=-1.,
    max_time_length=config.data.max_time_length,
    max_space_length=max_space_length,
    dataset_name=config.data.dataset_name,
    sort_by_depth=config.data.sort_by_depth,
    sort_by_region=config.data.sort_by_region,
    stitching=True,
    seed=config.seed,
    data_dir=f"{args.data_path}/{local_data_dir}" if not args.use_nlb else None,
    mode="test",
    eids=list(meta_data["eids"]) if args.num_sessions == 1 else None,
    shuffle=False,
    use_nlb=args.use_nlb
)

# --------
# SET PATH
# --------

num_sessions = len(meta_data["eid_list"])
if not args.use_nlb:
    eid_ = "multi" if num_sessions > 1 else eid[:5]
else:
    eid_ = eid

log_name = "sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_model-{}".format(
    num_sessions,
    eid_,
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    f"behavior-{'-'.join(avail_beh)}",
    model_class,
)

log_dir = os.path.join(base_path, "results", log_name)

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


# ------------
# SET UP MODEL
# ------------
if ("ap" in modal_filter["output"]) or ("spike" in modal_filter["output"]):
    if args.model == "linear":
        model_class = "LinearEncoder" 
    elif args.model == "rrr":
        model_class = "ReducedRankEncoder" 
        meta_data["rank"] = args.rank
    else:
        raise NotImplementedError
    
    input_size = n_beh if not args.use_nlb else sum([OUTPUT_DIM[beh] for beh in avail_beh])
    output_size = max_space_length
else: 
    if args.model == "linear":
        model_class = "LinearDecoder" 
    elif args.model == "rrr":
        model_class = "ReducedRankDecoder" 
        meta_data["rank"] = args.rank
    else:
        raise NotImplementedError
    
    input_size = max_space_length
    if ("choice" not in avail_beh) and ("block" not in avail_beh):
        output_size = n_beh if not args.use_nlb else sum([OUTPUT_DIM[beh] for beh in avail_beh])
    elif ("choice" in avail_beh) and n_beh == 1:
        output_size = 2
    elif ("block" in avail_beh) and n_beh == 1:
        output_size = 3

accelerator = Accelerator()

NAME2MODEL = {
    "LinearEncoder": BaselineEncoder, 
    "LinearDecoder": BaselineDecoder, 
    "ReducedRankEncoder": ReducedRankEncoder,
    "ReducedRankDecoder": ReducedRankDecoder,
}
model_class = NAME2MODEL[model_class]

if ("choice" in avail_beh) or ("block" in avail_beh):
    meta_data |= {"is_clf": True}
else:
    meta_data |= {"is_clf": False}

model = model_class(
    in_channel=input_size, 
    out_channel=output_size,
    seq_len=config.data.max_time_length,
    **config.method.model_kwargs, 
    **meta_data,
)

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

# -----
# TRAIN
# -----
trainer_kwargs = {
    "log_dir": log_dir,
    "accelerator": accelerator,
    "lr_scheduler": lr_scheduler,
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

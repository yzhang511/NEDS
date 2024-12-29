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
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding

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
ap.add_argument("--enc_task_var", type=str, default="all")
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


if args.multi_gpu:
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
else:
    accelerator = Accelerator()

max_lr = config.optimizer.lr
batch_size = config.training.train_batch_size
num_epochs = config.training.num_epochs
if args.multi_gpu:
    max_lr *= accelerator.num_processes
    num_epochs *= accelerator.num_processes

# ---------
# LOAD DATA
# ---------

train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
    config.dirs.dataset_cache_dir, 
    config.dirs.huggingface_org,
    num_sessions=args.num_sessions,
    eid = eid if args.num_sessions == 1 else None,
    use_re=True,
    split_method="predefined",
    test_session_eid=[],
    batch_size=batch_size,
    seed=config.seed
)

max_space_length = max(list(meta_data["eid_list"].values()))
logging.info(f"MAX space length to pad spike data to: {max_space_length}")

train_dataloader = make_loader(
    train_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
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
    data_dir=f"{args.data_path}/ibl_mm",
    mode="test",
    eids=list(meta_data["eids"]),
    shuffle=True
)

val_dataloader = make_loader(
    val_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
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
    data_dir=f"{args.data_path}/ibl_mm",
    mode="test",
    eids=list(meta_data["eids"]),
    shuffle=False
)

test_dataloader = make_loader(
    test_dataset, 
    target=[mod for mod in modality if mod in dynamic_acronyms],
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
    data_dir=f"{args.data_path}/ibl_mm",
    mode="test",
    eids=list(meta_data["eids"]),
    shuffle=False
)

# --------
# SET PATH
# --------

num_sessions = len(meta_data["eid_list"])
eid_ = "multi" if num_sessions > 1 else eid[:5]

log_name = \
"sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-{}".format(
    num_sessions,
    eid_, 
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    config.training.mask_type, 
    args.mask_mode,
    args.mask_ratio,
    args.enc_task_var,
)

log_dir = os.path.join(base_path, "results", log_name)

logging.info(f"Save model to {log_dir}")

final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, \
    "Last checkpoint exists and overwrite is False"
os.makedirs(log_dir, exist_ok=True)


# ------------
# SET UP MODEL
# ------------

logging.info(f"Start model training:")

if config.wandb.use:
    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb.project, 
            entity=config.wandb.entity, 
            config=config,
            name=log_name
        )


encoder_embeddings = {}

hidden_size = config.model.encoder.transformer.hidden_size
for mod in modal_filter["input"]:
    encoder_embeddings[mod] = EncoderEmbedding(
        hidden_size = hidden_size,
        n_channel = hidden_size,
        output_channel = hidden_size,
        stitching = True,
        eid_list = meta_data["eid_list"],
        mod = mod,
        config = config.model.encoder,
    )

NAME2MODEL = {"MultiModal": MultiModal}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(
    encoder_embeddings,
    avail_mod = neural_mods + static_mods + dynamic_mods,
    avail_beh = static_mods + dynamic_mods,
    model_mode = model_mode,
    config = config.model, 
    **config.method.model_kwargs, 
    **meta_data
)

optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=max_lr, 
        weight_decay=config.optimizer.wd, 
        eps=config.optimizer.eps
    )

grad_accum_steps = config.optimizer.gradient_accumulation_steps
global_batch_size = batch_size * accelerator.num_processes
total_steps=int(num_epochs*(len(train_dataset)//global_batch_size))//grad_accum_steps
if config.optimizer.scheduler == "linear":
    lr_scheduler = LinearLR(
        optimizer, 
        total_iters=total_steps
    )
elif config.optimizer.scheduler == "cosine":
    lr_scheduler = OneCycleLR(
        optimizer = optimizer,
        total_steps = total_steps,
        max_lr = config.optimizer.lr,
        pct_start = config.optimizer.warmup_pct,
        div_factor = config.optimizer.div_factor,
        anneal_strategy="cos",
    )

if args.continue_pretrain:

    best_pretrain_ckpt = "model_best_spike.pt"
    pretrain_path = \
    "sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-{}".format(
        num_sessions,
        "multi", 
        "-".join(modal_filter["input"]),
        "-".join(modal_filter["output"]),
        config.training.mask_type, 
        args.mask_mode,
        args.mask_ratio,
        args.enc_task_var,
    )
    pretrained_model_path = os.path.join(
        base_path, "results", pretrain_path, "pretrained", best_pretrain_ckpt
    )       

    model_state_dict = torch.load(pretrained_model_path)["model"]
    optimizer_state_dict = torch.load(pretrained_model_path)["optimizer"]
    lr_scheduler_state_dict = torch.load(pretrained_model_path)["lr_sched"]

    model = model.load_state_dict(model_state_dict)
    optimizer = optimizer.load_state_dict(optimizer_state_dict)
    lr_scheduler = lr_scheduler.load_state_dict(lr_scheduler_state_dict)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# -----------------------
# TRACK MODEL & DATA SIZE
# -----------------------

n_mods = len(modal_filter["input"])
n_tokens_per_mod = config.model.encoder.embedder.max_F
logging.info(f"Total modality: {n_mods} Total tokens per modality: {n_tokens_per_mod}")
logging.info(f"Total trials: {len(train_dataset)}")

total_tokens = n_mods*n_tokens_per_mod*len(train_dataset)
logging.info(f"Total tokens: {total_tokens}")

trial_length = 2 # Seconds
total_neurons = sum(list(meta_data["eid_list"].values()))
total_hours = len(train_dataset) * trial_length / 3_600
neuron_hours = total_neurons * total_hours
logging.info(f"Total neurons: {total_neurons}")
logging.info(f"Total hours: {total_hours}")
logging.info(f"Neuron hours: {neuron_hours}")

total_params = sum(p.numel() for p in model.parameters())
logging.info(f"Total parameters: {total_params}")

total_capacity = sum(
    p.numel() for name, p in model.named_parameters() 
    if "stitch" not in name and "static_weight" not in name
)
logging.info(f"Total parameters (excluding stitcher): {total_capacity}")


# -----
# TRAIN
# -----

trainer_kwargs = {
    "log_dir": log_dir,
    "accelerator": accelerator,
    "lr_scheduler": lr_scheduler,
    "avail_mod": neural_mods + static_mods + dynamic_mods,
    "avail_beh": static_mods + dynamic_mods,
    "modal_filter": modal_filter,
    "mixed_training": args.mixed_training,
    "enc_task_var": args.enc_task_var,
    "config": config,
    "multi_gpu": args.multi_gpu,
}

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
    logging.info(f"Starting dummy load with {args.dummy_size} samples")
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load, args.dummy_size))
    dummy_thread.start()
    try:
        trainer_.train()
    finally:
        stop_dummy_load.set()
        dummy_thread.join()
else:
    trainer_.train()


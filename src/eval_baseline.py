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
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.eval_baseline_utils import (
    load_model_data_local, 
    co_smoothing_eval
)

logging.basicConfig(level=logging.INFO) 

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel-speed", "whisker-motion-energy"]

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--model_mode", type=str, default="decoding")
ap.add_argument("--model", type=str, default="rrr", choices=["rrr", "linear"])
ap.add_argument("--behavior", nargs="+", default=["wheel-speed", "whisker-motion-energy"])
ap.add_argument("--modality", nargs="+", default=["ap", "behavior"])
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--finetune", action="store_true")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--save_plot", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

model_config = "src/configs/baseline.yaml"
if args.model_mode == "decoding":
    trainer_config = update_config("src/configs/trainer_decoder.yaml", model_config)
elif args.model_mode == "encoding":
    trainer_config = update_config("src/configs/trainer_encoder.yaml", model_config)
set_seed(args.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

# ------
# SET UP
# ------
eid = args.eid
base_path = args.base_path
avail_beh = args.behavior 
avail_mod = args.modality
model_mode = args.model_mode
model_class = args.model
n_beh = len(avail_beh)
num_sessions = args.num_sessions

if args.model_mode == "decoding":
    input_modal = ["ap"]
    output_modal = ["behavior"]
elif args.model_mode == "encoding":
    input_modal = ["behavior"]
    output_modal = ["ap"]
else:
    raise ValueError(f"Model mode {model_mode} not supported.")
    
modal_filter = {"input": input_modal, "output": output_modal}

# --------
# SET PATH
# --------

if num_sessions > 1:
    logging.warning("num_sessions > 1: ensure the model is trained with multiple sessions.")
    eid_ = "multi"
else:
    eid_ = eid[:5]

log_name = "sesNum-{}_ses-{}_set-eval_inModal-{}_outModal{}-model-{}".format(
    num_sessions,
    eid[:5], 
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    f"behavior-{'-'.join(avail_beh)}",
    model_class,
)

save_path = os.path.join(base_path, "results", log_name)

if args.finetune:
    pretrain_path = save_path.replace("eval", "finetune")
else:
    pretrain_path = save_path.replace("eval", "train")

logging.info(f"Save results to {save_path}")

if args.wandb:
    wandb.init(
        project="baseline",
        config=args,
        name=log_name
    )

# ----------
# LOAD MODEL
# ----------
configs = {
    "model_config": model_config,
    "model_path": pretrain_path,
    "trainer_config": trainer_config,
    "dataset_path": None, 
    "seed": args.seed,
    "eid": eid,
    "avail_mod": avail_mod,
    "avail_beh": avail_beh,
    "data_path": args.data_path,
}  
model, accelerator, dataset, dataloader = load_model_data_local(**configs)


# ----------
# EVAL MODEL
# ----------
modal_spike = True if model_mode == "encoding" else False
modal_behavior = True if model_mode == "decoding" else False

if modal_spike:
    modal_spike_bps_file = f"{save_path}/modal_spike/bps.npy"
    modal_spike_r2_file = f"{save_path}/modal_spike/r2.npy"
    if not os.path.exists(modal_spike_bps_file) or not os.path.exists(modal_spike_r2_file) or args.overwrite:
        logging.info(f"Start evaluation for encoding:")
        co_smoothing_configs = {
            "subtract": "task",
            "onset_alignment": [40],
            "save_path": f"{save_path}/eval_spike",
            "mode": "eval_spike",
            "n_time_steps": config.model.encoder.embedder.max_F,  
            "held_out_list": list(range(0, 100)),
            "is_aligned": True,
            "target_regions": None,
            "modal_filter": modal_filter,
            "target_to_decode": avail_beh, 
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        logging.info(results)
        wandb.log(results)
    else:
        logging.info("skipping modal_spike since files exist or overwrite is False")

if modal_behavior:
    modal_behavior_bps_file = f"{save_path}/modal_behavior/bps.npy"
    modal_behavior_r2_file = f"{save_path}/modal_behavior/r2.npy"
    if not os.path.exists(modal_behavior_bps_file) or not os.path.exists(modal_behavior_r2_file) or args.overwrite:
        logging.info(f"Start evaluation for decoding:")
        co_smoothing_configs = {
            "subtract": "task",
            "onset_alignment": [40],
            "save_path": f"{save_path}/eval_behavior",
            "mode": "eval_behavior",
            "n_time_steps": config.model.encoder.embedder.max_F,  
            "held_out_list": list(range(0, 100)),
            "is_aligned": True,
            "target_regions": None,
            "modal_filter": modal_filter,
            "target_to_decode": avail_beh, 
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        logging.info(results)
        wandb.log(results)
    else:
        logging.info("skipping modal_behavior since files exist or overwrite is False")

logging.info("Finish model evaluation")

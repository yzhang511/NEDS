import os
import wandb
import pickle
import logging
import argparse
import numpy as np
from math import ceil

import torch
from accelerate import Accelerator

from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import load_ibl_dataset
from utils.eval_utils import load_model_data_local, co_smoothing_eval

from multi_modal.mm import MultiModal

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
) 

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
ap.add_argument("--finetune", action="store_true")
ap.add_argument("--param_search", action="store_true")
ap.add_argument(
    "--modality", nargs="+", 
    default=["ap", "wheel-speed", "whisker-motion-energy", "choice", "block"]
)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--save_plot", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

if args.num_sessions == 1:
    model_config = f"src/configs/multi_modal/mm_single_session.yaml"
elif (args.num_sessions < 70) and (args.num_sessions > 10):
    model_config = f"src/configs/multi_modal/mm_medium_size.yaml"
elif args.num_sessions >= 70:
    model_config = f"src/configs/multi_modal/mm_large_size.yaml"
else:
    model_config = f"src/configs/multi_modal/mm.yaml" # default

kwargs = {"model": f"include:{model_config}"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/multi_modal/trainer_mm.yaml", config)
set_seed(config.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

neural_acronyms = {
    "ap": "spike", 
}
static_acronyms = {
    "choice": "choice", 
    "block": "block"
}
dynamic_acronyms = {
    "wheel-speed": "wheel", 
    "whisker-motion-energy": "whisker"
}


# ------ 
# SET UP
# ------
eid = args.eid
base_path = args.base_path
model_mode = args.model_mode
modality = list(neural_acronyms.keys()) + list(static_acronyms.keys()) + list(dynamic_acronyms.keys())
mask_mode = args.mask_mode
mask_name = f"mask_{mask_mode}"

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


# --------
# SET PATH
# --------
num_sessions = args.num_sessions
if num_sessions > 1:
    logging.warning("num_sessions > 1: ensure the model is trained with multiple sessions.")
    eid_ = "multi"
else:
    eid_ = eid[:5]

log_name = \
"sesNum-{}_ses-{}_set-eval_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-{}".format(
    num_sessions,
    eid[:5], 
    "-".join(modal_filter["input"]),
    "-".join(modal_filter["output"]),
    config.training.mask_type, 
    mask_mode,
    args.mask_ratio,
    args.enc_task_var,
)

save_path = os.path.join(base_path, "results", log_name)

if args.finetune:
    pretrain_path = save_path.replace("eval", "finetune")
else:
    pretrain_path = save_path.replace("eval", "train")

if args.param_search:
    log_name = f"{eid_}_{model_mode}"
    save_path = os.path.join(base_path, "results", log_name)
    tune_path = args.data_path.replace("datasets", f"tune/session_{num_sessions}/ray_results/{log_name}")
    tune_path = tune_path.replace("bcxj", "beez")
    print(f"Load ray tune model from: {tune_path}")
    if not os.path.exists(tune_path):
        tune_path = args.data_path.replace("datasets", f"ray_results/{log_name}")
    pretrain_path = [
        f for f in os.listdir(tune_path) if os.path.isdir(os.path.join(tune_path, f))
    ][0]
    pretrain_path = os.path.join(tune_path, pretrain_path)
    logging.info(f"Load best hyperparams model from {pretrain_path}.")
    with open(f"{pretrain_path}/params.pkl", "rb") as file:
        params = pickle.load(file)
    if not args.finetune:
        config["model"]["encoder"]["transformer"]["hidden_size"] = params["hidden_size"]
        config["model"]["encoder"]["transformer"]["inter_size"] = params["inter_size"]
        config["model"]["encoder"]["transformer"]["n_layers"] = params["n_layers"]
    model_config = config

logging.info(f"Save results to {save_path}.")

if args.wandb:
    wandb.init(
        project=config.wandb.project, 
        entity=config.wandb.entity, 
        config=args,
        name=log_name
    )

# ----------
# LOAD MODEL
# ----------
if args.model_mode == "mm":
    if args.enc_task_var in ["all", "random"]:
        best_ckpt_path = [
            "model_best_avg.pt", 
        ]
    else:
        best_ckpt_path = ["model_best_avg.pt"]
else:
    best_ckpt_path = ["model_best_avg.pt"]

avg_state_dict = []
for ckpt_path in best_ckpt_path:
    model_path = os.path.join(pretrain_path, ckpt_path)    
    configs = {
        "model_config": model_config,
        "model_path": model_path,
        "trainer_config": "src/configs/multi_modal/trainer_mm.yaml",
        "dataset_path": None, 
        "seed": 42,
        "mask_name": mask_name,
        "eid": eid,
        "neural_mods": neural_mods,
        "static_mods": static_mods,
        "dynamic_mods": dynamic_mods,
        "modal_filter": modal_filter,
        "model_mode": model_mode,
        "data_path": args.data_path,
    }      
    model, accelerator, dataset, dataloader = load_model_data_local(**configs)
    model_state_dict = model.state_dict()
    avg_state_dict.append(model_state_dict)

# Model Averaging
for key in model_state_dict:
    model_state_dict[key] = sum(
        [state_dict[key] for state_dict in avg_state_dict]
    ) / len(avg_state_dict)
model.load_state_dict(model_state_dict)


# ----------
# EVAL MODEL
# ----------
eval_spike = True if model_mode in ["mm", "encoding"] else False
eval_behavior = True if model_mode in ["mm", "decoding"] else False

logging.info(f"Start model evaluation:")

if eval_spike:
    eval_spike_bps_file = f"{save_path}/eval_spike/bps.npy"
    eval_spike_r2_file = f"{save_path}/eval_spike/r2.npy"
    if not os.path.exists(eval_spike_bps_file) or \
        not os.path.exists(eval_spike_r2_file) or args.overwrite:
        logging.info(f"Start evaluation for encoding:")
        co_smoothing_configs = {
            "subtract": "task",
            "onset_alignment": [40],
            "method_name": mask_name, 
            "save_path": f"{save_path}/eval_spike",
            "mode": "eval_spike",
            "n_time_steps": model.encoder_embeddings[modal_filter["input"][0]].max_F,  
            "held_out_list": list(range(model.encoder_embeddings[modal_filter["input"][0]].max_F)),
            "is_aligned": True,
            "target_regions": None,
            "enc_task_var": args.enc_task_var,
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            is_multimodal=True if model_mode == "mm" else False,
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        logging.info(results)
        wandb.log(results) if args.wandb else None
    else:
        logging.info("Skip evaluation for encoding since files exist or overwrite is False.")


if eval_behavior:
    eval_behavior_bps_file = f"{save_path}/eval_behavior/bps.npy"
    eval_behavior_r2_file = f"{save_path}/eval_behavior/r2.npy"
    if not os.path.exists(eval_behavior_bps_file) or \
        not os.path.exists(eval_behavior_r2_file) or args.overwrite:
        logging.info(f"Start evaluation for decoding:")
        co_smoothing_configs = {
            "subtract": "task",
            "onset_alignment": [40],
            "method_name": mask_name, 
            "save_path": f"{save_path}/eval_behavior",
            "mode": "eval_behavior",
            "n_time_steps": model.encoder_embeddings["spike"].max_F,  
            "held_out_list": list(range(model.encoder_embeddings["spike"].max_F)),
            "is_aligned": True,
            "target_regions": None,
            "avail_beh": static_mods + dynamic_mods,
        }
        results = co_smoothing_eval(
            model=model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            is_multimodal=True if model_mode == "mm" else False,
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        logging.info(results)
        wandb.log(results) if args.wandb else None
    else:
        logging.info("Skip evaluation for decoding since files exist or overwrite is False.")


if (args.enc_task_var == "random"):
    # Mask selected modalities for encoding
    for mod in static_mods + dynamic_mods:

        model_path = os.path.join(pretrain_path, f"model_best_enc_{mod}.pt")    
        configs = {
            "model_config": model_config,
            "model_path": model_path,
            "trainer_config": "src/configs/multi_modal/trainer_mm.yaml",
            "dataset_path": None, 
            "seed": 42,
            "mask_name": mask_name,
            "eid": eid,
            "neural_mods": neural_mods,
            "static_mods": static_mods,
            "dynamic_mods": dynamic_mods,
            "modal_filter": modal_filter,
            "model_mode": model_mode,
            "data_path": args.data_path,
        }      
        model, accelerator, dataset, dataloader = load_model_data_local(**configs)

        eval_spike_bps_file = f"{save_path}/eval_spike_{mod}/bps.npy"
        eval_spike_r2_file = f"{save_path}/eval_spike_{mod}/r2.npy"
        if not os.path.exists(eval_spike_bps_file) or \
            not os.path.exists(eval_spike_r2_file) or args.overwrite:
            logging.info(f"Start evaluation for encoding using {mod}:")
            co_smoothing_configs = {
                "subtract": "task",
                "onset_alignment": [40],
                "method_name": mask_name, 
                "save_path": f"{save_path}/eval_spike_{mod}",
                "mode": "eval_spike",
                "n_time_steps": model.encoder_embeddings["spike"].max_F,  
                "held_out_list": list(range(model.encoder_embeddings["spike"].max_F)),
                "is_aligned": True,
                "target_regions": None,
                "enc_task_var": mod,
            }
            results = co_smoothing_eval(
                model=model, 
                accelerator=accelerator, 
                test_dataloader=dataloader, 
                test_dataset=dataset, 
                is_multimodal=True if model_mode == "mm" else False,
                save_plot=args.save_plot,
                **co_smoothing_configs
            )
            logging.info(results)
            wandb.log(results) if args.wandb else None
        else:
            logging.info("Skip evaluation for encoding since files exist or overwrite is False.")

logging.info("Finish model evaluation")

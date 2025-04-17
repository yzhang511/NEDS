import os
import wandb
import pickle
import logging
import argparse
import numpy as np
from math import ceil
import torch
from utils.dataset_utils import load_ibl_dataset
from multi_modal.mm import MultiModal
from utils.utils import set_seed
from loader.make_loader import make_loader
from utils.config_utils import config_from_kwargs, update_config
from utils.eval_utils import (
    load_model_data_local, 
    co_smoothing_eval
)
from accelerate import Accelerator

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
) 

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
ap.add_argument("--config_dir", type=str, default="src/configs")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--save_plot", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

if args.num_sessions <= 10:
    model_config = f"src/configs/multi_modal/mm_single_session.yaml"
elif args.num_sessions == 40: 
    model_config = f"src/configs/multi_modal/mm_medium_size.yaml"
elif args.num_sessions == 74: 
    model_config = f"src/configs/multi_modal/mm_large_size.yaml"
else:
    model_config = f"src/configs/multi_modal/mm.yaml"

kwargs = {"model": f"include:{model_config}"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/multi_modal/trainer_mm.yaml", config)
set_seed(config.seed)

best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

# ------ 
# SET UP
# ------
eid = args.eid
base_path = args.base_path
model_mode = args.model_mode
modality = args.modality
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

finetune_path = save_path.replace("eval", "finetune")
pretrain_path = save_path.replace("eval", "train")
pretrain_path = pretrain_path.replace(eid[:5], "multi")

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

logging.info("Load pretrained model: ")


accelerator = Accelerator()

if args.model_mode == "mm":
    best_ckpt_path = [
        "model_best_avg.pt", 
    ]
else:
    best_ckpt_path = ["model_best_avg.pt"]

avg_state_dict = []
for ckpt_path in best_ckpt_path:
    model_path = os.path.join(pretrain_path, ckpt_path)    
    configs = {
        "model_config": model_config,
        "model_path": model_path,
        "trainer_config": f"{args.config_dir}/multi_modal/trainer_mm.yaml",
        "dataset_path": None, 
        "seed": 42,
        "mask_name": mask_name,
        "eid": eid,
        "neural_mods": neural_mods,
        "static_mods": static_mods,
        "dynamic_mods": dynamic_mods,
        "modal_filter": modal_filter,
        "model_mode": model_mode,
        "num_sessions": num_sessions,
        "data_path": args.data_path,
    }      
    pretrain_model, accelerator, dataset, dataloader = load_model_data_local(**configs)
    model_state_dict = pretrain_model.state_dict()
    avg_state_dict.append(model_state_dict)

# Model Averaging
for key in model_state_dict:
    model_state_dict[key] = sum(
        [state_dict[key] for state_dict in avg_state_dict]
    ) / len(avg_state_dict)
pretrain_model.load_state_dict(model_state_dict)

# -------------------
# LOAD FINETUNE MODEL
# -------------------

logging.info("Load finetuned model: ")

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
    model_path = os.path.join(finetune_path, ckpt_path)    
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
    finetune_model, accelerator, dataset, dataloader = load_model_data_local(**configs)
    model_state_dict = finetune_model.state_dict()
    avg_state_dict.append(model_state_dict)

# Model Averaging
for key in model_state_dict:
    model_state_dict[key] = sum(
        [state_dict[key] for state_dict in avg_state_dict]
    ) / len(avg_state_dict)
finetune_model.load_state_dict(model_state_dict)


# ----------
# EVAL MODEL
# ----------

pretrain_eids = list(
    pretrain_model.encoder_embeddings["spike"].mod_stitcher_proj_dict.stitch_decoder_dict.keys()
)

gt_choice, gt_block, pred_choice, pred_block = [], [], [], []
for idx, _eid in enumerate(pretrain_eids):

    print(f"Ensemble sub-model {idx} using EID {_eid}:")

    hidden_size = config.model.encoder.transformer.hidden_size

    for mod in static_mods + dynamic_mods:
        # pos_embed = pretrain_model.encoder_embeddings[mod].embedder.pos_embed.state_dict()
        # mod_emb = pretrain_model.encoder_embeddings[mod].embedder.mod_emb.state_dict()
        # session_emb = pretrain_model.encoder_embeddings[mod].embedder.session_emb.state_dict()
        
        if mod in static_mods:
            finetune_model.encoder_embeddings[mod].mod_static_weight_dict[eid] = pretrain_model.encoder_embeddings[mod].mod_static_weight_dict[_eid]
        finetune_model.encoder_embeddings[mod].mod_stitcher_proj_dict.stitch_decoder_dict[eid] = pretrain_model.encoder_embeddings[mod].mod_stitcher_proj_dict.stitch_decoder_dict[_eid]

        # finetune_model.encoder_embeddings[mod].embedder.pos_embed.load_state_dict(pos_embed)
        # finetune_model.encoder_embeddings[mod].embedder.mod_emb.load_state_dict(mod_emb)
        # finetune_model.encoder_embeddings[mod].embedder.session_emb.load_state_dict(session_emb)

    logging.info(f"Start model evaluation:")

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
            "n_time_steps": config.model.encoder.embedder.max_F,  
            "held_out_list": list(range(0, 100)),
            "is_aligned": True,
            "target_regions": None,
            "avail_beh": static_mods + dynamic_mods,
            "eval_zero_shot": True,
        }
        results = co_smoothing_eval(
            model=finetune_model, 
            accelerator=accelerator, 
            test_dataloader=dataloader, 
            test_dataset=dataset, 
            is_multimodal=True if model_mode == "mm" else False,
            save_plot=args.save_plot,
            **co_smoothing_configs
        )
        logging.info(results)
        wandb.log(results) if args.wandb else None

        gt_choice = results["gt_choice"]
        gt_block = results["gt_block"]
        pred_choice.append(results["pred_choice"])
        pred_block.append(results["pred_block"])
    else:
        logging.info("Skip evaluation for decoding since files exist or overwrite is False.")

logging.info("Finish model evaluation")

from sklearn.metrics import accuracy_score, balanced_accuracy_score

pred_choice = np.mean(pred_choice, 0)
pred_block = np.mean(pred_block, 0)

print(accuracy_score(gt_choice, (pred_choice > 0.5).astype(int)))
print(accuracy_score(gt_block, (pred_block > 0.5).astype(int)))

print(balanced_accuracy_score(gt_choice, (pred_choice > 0.5).astype(int)))
print(balanced_accuracy_score(gt_block, (pred_block > 0.5).astype(int)))



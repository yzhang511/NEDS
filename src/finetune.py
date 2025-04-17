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
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from collections import defaultdict
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from utils.eval_utils import load_model_data_local
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

def main(tune_config=None):

    if args.num_sessions <= 10:
        model_config = f"{args.config_dir}/multi_modal/mm_single_session.yaml"
    elif args.num_sessions == 40:
        model_config = f"{args.config_dir}/multi_modal/mm_medium_size.yaml"
    elif args.num_sessions == 74:
        model_config = f"{args.config_dir}/multi_modal/mm_large_size.yaml"
    else:
        model_config = f"{args.config_dir}/multi_modal/mm.yaml"

    kwargs = {"model": f"include:{model_config}"}
    config = config_from_kwargs(kwargs)
    config = update_config(f"{args.config_dir}/multi_modal/trainer_mm.yaml", config)
    set_seed(config.seed)

    best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

    # ------ 
    # SET UP
    # ------ 

    eid = args.eid
    base_path = args.base_path
    model_mode = args.model_mode
    modality = args.modality
    num_sessions = args.num_sessions
    if args.search:
        config["wandb"]["use"] = False
        config["model"]["masker"]["ratio"] = tune_config["mask_ratio"]
        lr = tune_config["learning_rate"]
        wd = tune_config["weight_decay"]
    else:
        config["model"]["masker"]["ratio"] = args.mask_ratio
        lr = config.optimizer.lr
        wd = config.optimizer.wd
    mask_mode = args.mask_mode
    mask_name = f"mask_{mask_mode}"
    config["model"]["masker"]["mode"] = args.mask_mode
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


    # ---------
    # LOAD DATA
    # ---------

    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
        args.data_path, 
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
        target=[mod for mod in modality if mod in dynamic_acronyms],
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
        data_dir=f"{args.data_path}/ibl_mm",
        mode="train",
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
        max_space_length=meta_data["num_neurons"][0],
        dataset_name=config.data.dataset_name,
        sort_by_depth=config.data.sort_by_depth,
        sort_by_region=config.data.sort_by_region,
        stitching=True,
        seed=config.seed,
        data_dir=f"{args.data_path}/ibl_mm",
        mode="val",
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
        max_space_length=meta_data["num_neurons"][0],
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
    num_sessions = args.num_sessions
    eid_ = "multi" if num_sessions > 1 else eid[:5]

    pretrain_path = \
    "sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-all".format(
        num_sessions,
        eid_, 
        "-".join(modal_filter["input"]),
        "-".join(modal_filter["output"]),
        config.training.mask_type, 
        args.mask_mode,
        args.mask_ratio,
        args.pretrain_task_var,
    )

    log_name = \
    "sesNum-{}_ses-{}_set-finetune_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-{}".format(
        num_sessions,
        eid[:5], 
        "-".join(modal_filter["input"]),
        "-".join(modal_filter["output"]),
        config.training.mask_type, 
        args.mask_mode,
        args.mask_ratio,
        args.enc_task_var,
    )

    if args.search:
        trial_dir = train.get_context().get_trial_dir()
        trial_name = os.path.basename(trial_dir)
        log_dir = os.path.join(ray_path, f"{eid[:5]}_{model_mode}", trial_name)
    else:
        log_dir = os.path.join(base_path, "results", log_name)

    logging.info(f"Save model to {log_dir}")

    final_checkpoint = os.path.join(log_dir, last_ckpt_path)
    assert not os.path.exists(final_checkpoint) or args.overwrite, \
        "Last checkpoint exists and overwrite is False"
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

    if args.model_mode == "mm":
        best_ckpt_path = [
            "model_best_avg.pt", 
            # "model_best_spike.pt",
            # "model_best_wheel.pt", 
            # "model_best_whisker.pt",
            # "model_best_choice.pt",
            # "model_best_block.pt"
        ]
    else:
        best_ckpt_path = ["model_best_avg.pt"]

    avg_state_dict = []
    for ckpt_path in best_ckpt_path:
        model_path = os.path.join(
            base_path, "results", pretrain_path, ckpt_path
        )    
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
        model, accelerator, dataset, dataloader = load_model_data_local(**configs)
        model_state_dict = model.state_dict()
        avg_state_dict.append(model_state_dict)

    # Model Averaging
    for key in model_state_dict:
        model_state_dict[key] = sum(
            [state_dict[key] for state_dict in avg_state_dict]
        ) / len(avg_state_dict)
    model.load_state_dict(model_state_dict)

    model.masker.ratio = args.mask_ratio
    logging.info(f"Reset mask ratio to {model.masker.ratio} for fine-tuning.")

    # -----------------------
    # ACCOMMODATE NEW SESSION
    # -----------------------

    if num_sessions > 1:

        hidden_size = config.model.encoder.transformer.hidden_size

        for mod in neural_mods + static_mods + dynamic_mods:
            pos_embed = model.encoder_embeddings[mod].embedder.pos_embed.state_dict()
            mod_emb = model.encoder_embeddings[mod].embedder.mod_emb.state_dict()
            session_emb = model.encoder_embeddings[mod].embedder.session_emb.state_dict()
            model.encoder_embeddings[mod] = EncoderEmbedding(
                hidden_size = hidden_size,
                n_channel = hidden_size,
                output_channel = hidden_size,
                stitching = True,
                eid_list = meta_data["eid_list"],
                mod = mod,
                config = config.model.encoder,
            )
            model.encoder_embeddings[mod].embedder.pos_embed.load_state_dict(pos_embed)
            model.encoder_embeddings[mod].embedder.mod_emb.load_state_dict(mod_emb)
            model.encoder_embeddings[mod].embedder.session_emb.load_state_dict(session_emb)

            if args.zero_shot_transfer:
                config["training"]["num_epochs"] = 500
                model.encoder_embeddings[mod].embedder.pos_embed.requires_grad = False
                model.encoder_embeddings[mod].embedder.mod_emb.requires_grad = False
                model.encoder_embeddings[mod].embedder.session_emb.requires_grad = False


    # -----------------------
    # TRACK MODEL & DATA SIZE
    # -----------------------

    if args.zero_shot_transfer:
        input_mods = output_mods = neural_mods
        modal_filter = {"input": input_mods, "output": output_mods}

    n_mods = len(modal_filter["input"])
    n_tokens_per_mod = config.model.encoder.embedder.max_F
    n_batches = len(train_dataloader)
    batch_size = config.training.train_batch_size
    logging.info(f"Total modality: {n_mods} Total tokens per modality: {n_tokens_per_mod}")
    logging.info(f"Total batch: {n_batches} batch size: {batch_size}")

    total_tokens = n_mods*n_tokens_per_mod*n_batches*batch_size
    logging.info(f"Total tokens: {total_tokens}")

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")

    # ------------
    # SET UP MODEL
    # ------------

    model = accelerator.prepare(model)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=wd, 
        eps=config.optimizer.eps
    )

    grad_accum_steps = config.optimizer.gradient_accumulation_steps

    lr_scheduler = OneCycleLR(
        optimizer = optimizer,
        total_steps = config.training.num_epochs*len(train_dataloader)//grad_accum_steps,
        max_lr = config.optimizer.lr,
        pct_start = config.optimizer.warmup_pct,
        div_factor = config.optimizer.div_factor,
    )

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
        "zero_shot_transfer": args.zero_shot_transfer,
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
            validation_metrics = trainer_.train()
        finally:
            stop_dummy_load.set()
            dummy_thread.join()
    else:
        validation_metrics = trainer_.train()
    train.report(validation_metrics)

    
if __name__ == "__main__":
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
    ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
    ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
    ap.add_argument("--num_sessions", type=int, default=1)
    ap.add_argument("--model_mode", type=str, default="mm")
    ap.add_argument("--mask_mode", type=str, default="temporal")
    ap.add_argument("--mask_ratio", type=float, default=0.1)
    ap.add_argument("--mixed_training", action="store_true")
    ap.add_argument("--pretrain_task_var", type=str, default="random")
    ap.add_argument("--enc_task_var", type=str, default="all")
    ap.add_argument(
        "--modality", nargs="+", 
        default=["ap", "wheel-speed", "whisker-motion-energy", "choice", "block"]
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dummy_load", action="store_true")
    ap.add_argument("--dummy_size", type=int, default=50000)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--zero_shot_transfer", action="store_true")
    ap.add_argument("--num_tune_sample", type=int, default=10)
    ap.add_argument("--config_dir", type=str, default="src/configs")
    args = ap.parse_args()

    if args.search:
        ray.init(address="auto")  
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(0.001, 0.1),
            "mask_ratio": tune.uniform(0.1, 0.4),
        }
        ray_path = os.path.join(args.base_path, "ray_results")
        scheduler = ASHAScheduler(
            metric="eval_avg_metric",
            mode="max",
            grace_period=1,
            reduction_factor=2
        )
        print("Starting hyperparameter search")
        print(f"saving to {ray_path}")

        eid_ = args.eid[:5]
        
        analysis = tune.run(
            main,
            resources_per_trial={
                "cpu": 1,
                "gpu": 1  
            },
            config=search_space,
            num_samples=args.num_tune_sample,
            scheduler=scheduler,
            storage_path=ray_path,
            name=f"{eid_}_{args.model_mode}",
            log_to_file=True,
            verbose=2
        )
        # Get the best hyperparameters
        best_hyperparameters = analysis.get_best_config(
            metric="eval_avg_metric",
            mode="max"
        )
        logging.info(f"Best hyperparameters: {best_hyperparameters}")
    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        logging.info(f"No hyperparameter search, Starting training")
        main()

        
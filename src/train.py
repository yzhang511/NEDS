import os
import wandb
import pickle
import logging
import argparse
import threading
import numpy as np

import torch
from torch.optim.lr_scheduler import OneCycleLR, LinearLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

from utils.utils import set_seed, dummy_load
from utils.dataset_utils import load_ibl_dataset
from utils.config_utils import config_from_kwargs, update_config

from loader.make_loader import make_loader
from trainer.make import make_multimodal_trainer

from multi_modal.mm import MultiModal
from multi_modal.encoder_embeddings import EncoderEmbedding


def main(tune_config=None):

    neural_acronyms = {
        "ap": "spike"
    }
    static_acronyms = {
        "choice": "choice", 
        "block": "block"
    }
    dynamic_acronyms = {
        "wheel-speed": "wheel", 
        "whisker-motion-energy": "whisker"
    }

    if args.num_sessions == 1:
        model_config = f"{args.config_dir}/multi_modal/mm_single_session.yaml"
    elif (args.num_sessions < 70) and (args.num_sessions > 10):
        model_config = f"{args.config_dir}/multi_modal/mm_medium_size.yaml"
    elif args.num_sessions >= 70:
        model_config = f"{args.config_dir}/multi_modal/mm_large_size.yaml"
    else:
        model_config = f"{args.config_dir}/multi_modal/mm.yaml" # default

    kwargs = {"model": f"include:{model_config}"}
    config = config_from_kwargs(kwargs)
    
    if args.num_sessions <= 40:
        config = update_config(f"{args.config_dir}/multi_modal/trainer_mm.yaml", config)
    else:
        config = update_config(f"{args.config_dir}/multi_modal/trainer_multi_session.yaml", config)
        
    if args.model_mode == "encoding":
        config["training"]["num_epochs"] = 4000

    set_seed(config.seed)

    best_ckpt_path, last_ckpt_path = "model_best.pt", "model_last.pt"

    # ------ 
    # SET UP
    # ------ 
    eid = args.eid
    base_path = args.base_path
    model_mode = args.model_mode
    modality = list(neural_acronyms.keys()) + list(static_acronyms.keys()) + list(dynamic_acronyms.keys())

    if args.search:
        config["wandb"]["use"] = False
        mask_ratio = tune_config["mask_ratio"]
        lr = tune_config["learning_rate"]
        wd = tune_config["weight_decay"]
        hidden_size = tune_config["hidden_size"]
        inter_size = tune_config["inter_size"]
        n_layers = tune_config["n_layers"]
        config["model"]["encoder"]["transformer"]["hidden_size"] = hidden_size
        config["model"]["encoder"]["transformer"]["inter_size"] = inter_size
        config["model"]["encoder"]["transformer"]["n_layers"] = n_layers
    else:
        mask_ratio = args.mask_ratio
        lr = config.optimizer.lr
        wd = config.optimizer.wd
        hidden_size = config.model.encoder.transformer.hidden_size
        inter_size = config.model.encoder.transformer.inter_size

    logging.info(f"EID: {eid} model mode: {args.model_mode} mask ratio: {mask_ratio}")
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

    max_num_processes = 30

    batch_size = config.training.train_batch_size
    global_batch_size = batch_size
    num_epochs = 4_000 if model_mode == "encoding" else config.training.num_epochs
    max_lr = 5e-4 if model_mode == "encoding" else lr

    if args.multi_gpu:
        num_epochs *= accelerator.num_processes
        if accelerator.num_processes > max_num_processes:
            max_lr *= max_num_processes / 2
            global_batch_size = 512
        elif args.num_sessions >= 70:
            max_lr = 0.003
            global_batch_size = 1024
        else:
            max_lr *= accelerator.num_processes
            global_batch_size *= accelerator.num_processes 

    # ---------
    # LOAD DATA
    # ---------
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
        args.data_path, 
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

    local_data_dir = "ibl_mm" if args.num_sessions == 1 else f"ibl_mm_{args.num_sessions}"

    target_behavior_lst = [mod for mod in modality if mod in dynamic_acronyms]

    train_dataloader = make_loader(
        train_dataset, 
        target=target_behavior_lst,
        load_meta=config.data.load_meta,
        batch_size=batch_size, 
        pad_to_right=True, 
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        sort_by_depth=config.data.sort_by_depth,
        sort_by_region=config.data.sort_by_region,
        stitching=True,
        seed=config.seed,
        data_dir=f"{args.data_path}/{local_data_dir}",
        mode="train",
        eids=list(meta_data["eids"]),
        shuffle=True,
    )
    val_dataloader = make_loader(
        val_dataset, 
        target=target_behavior_lst,
        load_meta=config.data.load_meta,
        batch_size=batch_size, 
        pad_to_right=True, 
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        sort_by_depth=config.data.sort_by_depth,
        sort_by_region=config.data.sort_by_region,
        stitching=True,
        seed=config.seed,
        data_dir=f"{args.data_path}/{local_data_dir}",
        mode="val",
        eids=list(meta_data["eids"]),
        shuffle=False,
    )
    test_dataloader = make_loader(
        test_dataset, 
        target=target_behavior_lst,
        load_meta=config.data.load_meta,
        batch_size=batch_size, 
        pad_to_right=True, 
        pad_value=-1.,
        max_time_length=config.data.max_time_length,
        max_space_length=max_space_length,
        dataset_name=config.data.dataset_name,
        sort_by_depth=config.data.sort_by_depth,
        sort_by_region=config.data.sort_by_region,
        stitching=True,
        seed=config.seed,
        data_dir=f"{args.data_path}/{local_data_dir}",
        mode="test",
        eids=list(meta_data["eids"]),
        shuffle=False,
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
        mask_ratio,
        args.enc_task_var,
    )
    if args.search:
        trial_dir = train.get_context().get_trial_dir()
        trial_name = os.path.basename(trial_dir)
        log_dir = os.path.join(ray_path, f"{eid_}_{model_mode}", trial_name)
    else: 
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
            max_F = config.data.max_time_length,
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
        weight_decay=wd, 
        eps=config.optimizer.eps
    )

    num_train = len(train_dataset["eid"])
    grad_accum_steps = config.optimizer.gradient_accumulation_steps
    total_steps=int(num_epochs*(num_train//global_batch_size))//grad_accum_steps
    if config.optimizer.scheduler == "linear":
        lr_scheduler = LinearLR(
            optimizer, 
            total_iters=total_steps
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = OneCycleLR(
            optimizer = optimizer,
            total_steps = total_steps,
            max_lr = max_lr,
            pct_start = config.optimizer.warmup_pct,
            div_factor = config.optimizer.div_factor,
            anneal_strategy="cos",
        )

    if args.continue_pretrain:
        best_pretrain_ckpt = "model_epoch.pt"
        pretrain_path = \
        "sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_taskVar-{}".format(
            num_sessions,
            "multi", 
            "-".join(modal_filter["input"]),
            "-".join(modal_filter["output"]),
            config.training.mask_type, 
            args.mask_mode,
            mask_ratio,
            args.enc_task_var,
        )
        pretrained_model_path = os.path.join(
            base_path, "results", pretrain_path, "pretrained", best_pretrain_ckpt
        )       

        checkpoint = torch.load(pretrained_model_path)
        model_state_dict = checkpoint["model"]
        optimizer_state_dict = checkpoint["optimizer"]
        lr_scheduler_state_dict = checkpoint["lr_sched"]
        start_epoch = checkpoint["epoch"] + 1
    
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        print(f"Resume training from epoch {start_epoch}.")
    else:
        start_epoch = 0

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -----------------------
    # TRACK MODEL & DATA SIZE
    # -----------------------
    n_mods = len(modal_filter["input"])
    n_tokens_per_mod = config.model.encoder.embedder.max_F
    num_train = len(train_dataset["eid"])
    logging.info(f"Total modality: {n_mods} Total tokens per modality: {n_tokens_per_mod}")
    logging.info(f"Total trials: {num_train}")

    total_tokens = n_mods*n_tokens_per_mod*num_train
    logging.info(f"Total tokens: {total_tokens}")

    trial_length = 2 # Seconds
    total_neurons = sum(list(meta_data["eid_list"].values()))
    total_hours = num_train * trial_length / 3_600
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
        "start_epoch": start_epoch, 
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
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--num_tune_sample", type=int, default=50)
    ap.add_argument("--config_dir", type=str, default="configs")
    args = ap.parse_args()

    if args.debug:
        # Debug using deterministic mode
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logging.info("Deterministic mode is activated. This will negatively impact performance.")
        
    if args.search:
        ray.init(address="auto")  
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(0.001, 0.1),
            "mask_ratio": tune.uniform(0.1, 0.4),
            "hidden_size": tune.choice([128, 256, 512]),
            "inter_size": tune.choice([256, 512, 1024]),
            "n_layers": tune.choice([5, 6]),
        }
        ray_path = os.path.join(args.base_path, "ray_results")
        scheduler = ASHAScheduler(
            metric="eval_avg_metric",
            mode="max",
            grace_period=1,
            reduction_factor=2
        )
        print("Starting hyperparameter search")
        print(f"Saving to {ray_path}")

        eid_ = "multi" if args.num_sessions > 1 else args.eid[:5]
        
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
        
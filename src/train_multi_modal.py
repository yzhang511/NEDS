import os
import pickle
import argparse
from math import ceil
import numpy as np
import torch
import wandb
import warnings
import threading

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset, split_both_dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from multi_modal.mm import MultiModal
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_multimodal_trainer
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.decoder_embeddings import DecoderEmbedding


ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='db4df448-e449-4a6f-a0e7-288711e7a75a')
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--use_MtM", action='store_true')
ap.add_argument("--mixed_training", action='store_true')
ap.add_argument("--overwrite", action='store_true')
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/yzhang39/temp_project")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--dummy_load", action='store_true')
ap.add_argument("--dummy_size", type=int, default=50000)

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
    "input": ['ap', 'behavior'], 
    "output": ['ap', 'behavior']
}
eid_ = args.eid if args.num_sessions == 1 else None
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                                    config.dirs.huggingface_org,
                                    num_sessions=args.num_sessions,
                                    eid = eid_,
                                    use_re=True,
                                    split_method="predefined",
                                    test_session_eid=[],
                                    batch_size=config.training.train_batch_size,
                                    seed=config.seed)

num_sessions = len(meta_data['eid_list'])
mask_mode = '-'.join(config.training.mask_mode) if config.training.mask_type == 'input' else args.mask_mode
eid_ = "multi" if num_sessions > 1 else eid[:5]

log_dir = os.path.join(base_path, 
                       "results",
                       f"sesNum-{num_sessions}",
                       f"ses-{eid_}",
                       "set-train",
                       f"inModal-{'-'.join(modal_filter['input'])}",
                       f"outModal-{'-'.join(modal_filter['output'])}",
                       f"mask-{config.training.mask_type}",
                       f"mode-{mask_mode}",
                       f"ratio-{args.mask_ratio}",
                       f"mixedTraining-{args.mixed_training}"
                       )
final_checkpoint = os.path.join(log_dir, last_ckpt_path)
assert not os.path.exists(final_checkpoint) or args.overwrite, "last checkpoint exists and overwrite is False"
os.makedirs(log_dir, exist_ok=True)
if config.wandb.use:
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, config=config,
        name="sesNum-{}_ses-{}_set-train_inModal-{}_outModal-{}_mask-{}_mode-{}_ratio-{}_mixedTraining-{}".format(
            num_sessions,
            eid_, 
            '-'.join(modal_filter['input']),
            '-'.join(modal_filter['output']),
            config.training.mask_type, 
            mask_mode,
            args.mask_ratio,
            args.mixed_training
        )
    )

print('Start model training.')
print('=====================')

# dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir)
# train_dataset = dataset["train"]
# val_dataset = dataset["val"]
# test_dataset = dataset["test"]
# print(dataset.column_names)

n_behaviors = len(avail_beh)
# n_neurons = len(train_dataset['cluster_regions'][0])
# meta_data['num_neurons'] = [n_neurons]
# print(meta_data)

train_dataloader = make_loader(train_dataset, 
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
                            shuffle=True)

val_dataloader = make_loader(val_dataset, 
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
                            shuffle=False)

test_dataloader = make_loader(test_dataset, 
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
                            shuffle=False)

encoder_embeddings, decoder_embeddings = {}, {}

standard_channel_size =256
for mod in modal_filter["input"]:
    encoder_embeddings[mod] = EncoderEmbedding(
        hidden_size=config.model.encoder.transformer.hidden_size,
        n_channel=standard_channel_size if mod == 'ap' else n_behaviors,
        stitching=True,
        eid_list=meta_data['eid_list'],
        config=config.model.encoder,
    )

for mod in modal_filter["output"]:
    decoder_embeddings[mod] = DecoderEmbedding(
        hidden_size=config.model.decoder.transformer.hidden_size,
        n_channel=standard_channel_size if mod == 'ap' else n_behaviors,
        output_channel=standard_channel_size if mod == 'ap' else n_behaviors,
        stitching=True,
        eid_list=meta_data['eid_list'],
        config=config.model.decoder,
    )

accelerator = Accelerator()

NAME2MODEL = {"MultiModal": MultiModal}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(
    encoder_embeddings,
    decoder_embeddings,
    avail_mod=avail_mod,
    config=config.model, 
    share_modality_embeddings=True,
    **config.method.model_kwargs, 
    **meta_data
)

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

# Shared variable to signal the dummy load to stop
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
    # Start the dummy load
    print(f"Starting dummy load with {args.dummy_size} samples")
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load, args.dummy_size))
    dummy_thread.start()

    try:
        trainer_.train()
    finally:
        stop_dummy_load.set()
        dummy_thread.join()
else:
    trainer_.train()

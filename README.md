# NEDS: Neural Encoding and Decoding at Scale

We introduce a multimodal, multi-task model that enables simultaneous [Neural Encoding and Decoding at Scale (NEDS)](https://arxiv.org/abs/2504.08201). Central to our approach is a novel multi-task-masking strategy, which alternates between neural, behavioral, within-modality, and cross-modality masking. 

![NEDS](assets/neds_schematic.png)

## Installation

```bash
conda env create -f env.yaml    # Create conda environment

conda activate neds             # Activate conda environment
```

## Datasets and Models

Download and prepare the IBL dataset. Update `base_path` and `data_path` in the scripts:

```bash
sbatch prepare_data.sh 1 EID    # Download data for a session using the provided EID

sbatch prepare_data.sh 84       # Download 84 sessions from the IBL repeated-site dataset
```

To accelerate data loading during training, we pre-save the partitioned train/val/test data and load it when needed:

```bash
sbatch create_dataset.sh 1 EID  # Save the train/val/test data for a session using the provided EID

source run_create_dataset.sh    # Save the train/val/test data for each of the 10 test sessions individually

sbatch create_dataset.sh 10     # Save the train/val/test data for pretraining on a set of 10 sessions
```

### Train NEDS

Train NEDS from scratch on a single session using a single GPU. Update `base_path` and `data_path` in the script:

```bash
sbatch train.sh 1 EID train mm 0 0.1 False random        # Train multi-modal model

sbatch train.sh 1 EID train encoding 0 0.1 False random  # Train encoding model

sbatch train.sh 1 EID train decoding 0 0.1 False all     # Train decoding model
```

Use `Ray Tune` for hyperparameter search. Update `num_tune_sample` in the script to change the number of random models:

```bash
sbatch train.sh 1 EID train mm 0 0.1 True random
```

Pre-train NEDS on 10 sessions using multiple GPUs. Update `--nodes` and `--ntasks` in the script to change the number of GPUs used:

```bash
sbatch train_multi_gpu.sh 10 none mm 0 0.1 all   # Pre-training requires "all"
```

Fine-tune the pre-trained 10-session NEDS model on a single held-out test session:

```bash
sbatch train.sh 10 EID finetune mm 0 0.1 False random
```

### Evaluate NEDS

To evaluate NEDS on a single session:

```bash
sbatch eval.sh 1 EID train mm 0.1 random False     # Eval a model trained from scratch (no hyperparameter search)

sbatch eval.sh 1 EID finetune mm 0.1 random False  # Eval a model fine-tuned on a held-out test session (no hyperparameter search)

sbatch eval.sh 1 EID train mm 0.1 random True      # Eval a model trained from scratch (with hyperparameter search)
```
**NOTE**: If you want to evaluate a model after hyperparameter tuning, locate the best model in the training logs and keep only its folder in `/YOUR_PATH/tune/`. Alternatively, you can change `eval.py` to automatically load the best checkpoint.


### Pretrained Weights

| Model | Description | Download |
|-------|-------------|------|
| NEDS (Medium) | NEDS trained on 40 sessions from IBL repeated site dataset | Coming soon |
| NEDS (Large) | NEDS trained on 79 sessions from IBL repeated site dataset | Coming soon |

The IBL sessions used for pre-training are in `data/train_eids.txt`, while the held-out sessions for evaluation are in `data/test_eids.txt`.


## Citation
Please cite our paper if you use this code in your own work:
```
@article{zhang2025neural,
  title={Neural Encoding and Decoding at Scale},
  author={Zhang, Yizi and Wang, Yanchen and Azabou, Mehdi and Andre, Alexandre and Wang, Zixuan and Lyu, Hanrui and Laboratory, The International Brain and Dyer, Eva and Paninski, Liam and Hurwitz, Cole},
  journal={arXiv preprint arXiv:2504.08201},
  year={2025}
}
```



# NEDS: Neural Encoding and Decoding at Scale

We introduce a multimodal, multi-task model that enables simultaneous [Neural Encoding and Decoding at Scale (NEDS)](https://arxiv.org/abs/2504.08201). Central to our approach is a novel multi-task-masking strategy, which alternates between neural, behavioral, within-modality, and cross-modality masking. 

![NEDS](assets/neds_schematic.png)

## Installation

```bash
conda env create -f env.yaml

conda activate neds
```

## Datasets and Models

Download and prepare the IBL data. Please change the `base_path` and `data_path` in the scripts to your own path:

```bash
sbatch prepare_data.sh 1 EID  # Download the data for a single session using a given EID
sbatch prepare_data.sh 84     # Download the sessions from the IBL repeated site dataset
```

To speed up data loading, we save the partitioned data ahead of time and load them later:
```bash
sbatch create_dataset.sh 1 EID  # Save the partitioned data for a single session  
source run_create_dataset.sh    # Save the partitioned data for the 10 test sessions (single session)
sbatch create_dataset.sh 10     # Save the partitioned data for pretraining on 10 sessions
```

### Train NEDS

To train NEDS (from scratch) on a single session using a single GPU:

```bash
bash script/train.sh 1 0 mm 100 False 0.5 False
```

To train NEDS (from scratch) on a single session using hyperparameter search:

```bash
bash script/train.sh 1 0 mm 100 True 0.5 False
```

To train NEDS (from scratch) on multiple sessions using multiple GPUs:

```bash
bash script/train.sh 1 0 mm 100 False 0.5 False
```

### Finetune NEDS

To finetune pretrained NEDS on a single session:

```bash
bash script/finetune.sh 1 0 mm 100 False 0.5 False
```

### Evaluate NEDS

To evaluate NEDS on a single session:

```bash
bash script/eval.sh 1 0 mm 100 False False
```

### Pretrained Weights

| Model | Description | Download |
|-------|-------------|------|
| NEDS (Medium) | NEDS trained on 40 sessions from IBL repeated site dataset | Coming soon |
| NEDS (Large) | NEDS trained on 79 sessions from IBL repeated site dataset | Coming soon |


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



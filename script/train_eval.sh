#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train_eval"
#SBATCH --output="train_eval.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-5
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../

python src/train_multi_modal.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a --base_path /scratch/bcxj/yzhang39 --mask_ratio 0.1 --mixed_training --num_sessions 1 --overwrite --model_mode mm

python src/eval_multi_modal.py --mask_mode temporal --mask_ratio 0.1 --eid db4df448-e449-4a6f-a0e7-288711e7a75a --seed 42 --base_path /scratch/bcxj/yzhang39 --save_plot --mask_type embd --mixed_training --num_sessions 1 --overwrite --model_mode mm --wandb

conda deactivate
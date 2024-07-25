#!/bin/bash

#SBATCH --job-name=eval-mm
#SBATCH --output=eval-mm-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

num_sessions=${1}
eid=${2}
model_mode=${3}
mask_rartio=${4}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../..

python src/eval_multi_modal.py --mask_mode temporal \
                               --mask_ratio ${mask_rartio} \
                               --eid ${eid} \
                               --seed 42 \
                               --base_path ./ \
                               --save_plot \
                               --mask_type embd \
                               --mixed_training  \
                               --num_sessions ${num_sessions} \
                               --model_mode ${model_mode} \
                               --wandb
cd script/ppwang

conda deactivate
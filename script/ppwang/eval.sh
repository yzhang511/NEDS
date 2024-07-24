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
mask_rartio=${3}

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
                               --wandb
cd script/ppwang

conda deactivate
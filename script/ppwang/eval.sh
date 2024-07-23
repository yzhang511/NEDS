#!/bin/bash

#SBATCH --job-name=ibl-fm
#SBATCH --output=ibl-fm.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

num_sessions=${1}
mask_rartio=${2}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../..

python src/eval_multi_modal.py --mask_mode temporal \
                               --mask_ratio ${mask_rartio} \
                               --eid 3638d102-e8b6-4230-8742-e548cd87a949 \
                               --seed 42 \
                               --base_path ./ \
                               --save_plot \
                               --mask_type embd \
                               --mixed_training  \
                               --num_sessions ${num_sessions} \
                               --wandb
cd script/ppwang

conda deactivate
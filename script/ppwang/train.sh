#!/bin/bash

#SBATCH --job-name=train-mm
#SBATCH --output=train-mm-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
num_sessions=${1}
eid=${2}
mask_ratio=${3}
echo $TMPDIR
conda activate ibl-mm

cd ../..

python src/train_multi_modal.py --eid $eid \
                                     --base_path ./ \
                                     --mask_ratio $mask_ratio \
                                     --mixed_training \
                                     --num_sessions $num_sessions \
                                     --dummy_size 130000 \
                                     --dummy_load

cd script/ppwang

conda deactivate
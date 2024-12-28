#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="create-dataset"
#SBATCH --output="create-dataset.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH -t 00:15:00
#SBATCH --gpus=1
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
eid=${1}
model_mode=mm
dummy_size=0
mask_ratio=0.1
echo $TMPDIR
conda activate ibl-mm

cd ../../..

python src/create_dataset.py --eid $eid \
                                --base_path ./\
                                --mask_ratio $mask_ratio \
                                --mixed_training \
                                --num_sessions 1 \
                                --dummy_size $dummy_size \
                                --model_mode $model_mode 

cd script/ppwang/expanse

conda deactivate

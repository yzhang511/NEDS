#!/bin/bash

#SBATCH --job-name=train-glvm
#SBATCH --output=train-glvmout
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
mask_ratio=${1}
echo $TMPDIR
conda activate ibl-mm

cd ../..

python src/train_glvm.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a \
                                     --base_path ./ 
                                    #  --mixed_training

cd script/ppwang

conda deactivate
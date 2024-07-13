#!/bin/bash

#SBATCH --job-name=train-baseline
#SBATCH --output=train-baseline.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g


. ~/.bashrc
echo $TMPDIR

conda activate ibl-mm

cd ../../

python src/train_baseline.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a \
                             --base_path ./ \
                             --overwrite

conda deactivate
cd script/ppwang


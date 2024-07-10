#!/bin/bash

#SBATCH --job-name=ibl-mm
#SBATCH --output=ibl-mm.out
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

cd ../..

python src/train_multi_modal.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a \
                                     --base_path ./ \
                                     --mask_ratio 0.3 

cd script/ppwang

conda deactivate
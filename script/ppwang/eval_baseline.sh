#!/bin/bash

#SBATCH --job-name=eval-baseline
#SBATCH --output=eval-baseline-%j.out
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

python src/eval_baseline.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a \
                            --seed 42 \
                            --base_path ./ \
                            --save_plot \
                            --overwrite \
                            --wandb

conda deactivate
cd script/ppwang

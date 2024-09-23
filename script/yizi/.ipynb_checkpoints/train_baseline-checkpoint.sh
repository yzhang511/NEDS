#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="baseline"
#SBATCH --output="baseline.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-06
#SBATCH --export=ALL

module load gpu
module load slurm


. ~/.bashrc
echo $TMPDIR

conda activate ibl-mm

cd ../../

python src/train_baseline.py --eid 824cf03d-4012-4ab1-b499-c83a92c5589e \
                             --base_path ./ \
                             --overwrite

conda deactivate
cd script/ppwang


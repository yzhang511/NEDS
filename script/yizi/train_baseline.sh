#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="rrr"
#SBATCH --output="rrr.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc

eid=${1}
model_mode=${2}
behavior=${3}

conda activate ibl-mm

cd ../../

if [ $behavior = "continuous" ]; then
    python src/train_baseline.py --eid $eid \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode
elif [ $behavior = "choice" ] || [ $behavior = "block" ];
then
    python src/train_baseline.py --eid $eid \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior $behavior
else
    echo "behavior: $behavior not supported"
fi

conda deactivate
cd script/yizi


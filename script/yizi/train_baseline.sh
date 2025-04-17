#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --job-name="rrr"
#SBATCH --output="rrr.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-02
#SBATCH --export=ALL

. ~/.bashrc

eid=${1}
model_mode=${2}
behavior=${3}
use_nlb=${4}

if [ "$use_nlb" = "True" ]; then
    echo "Using NLB"
    use_nlb="--use_nlb"
else
    echo "Not using NLB"
    search=""
fi

conda activate ibl-mm

cd ../../

if [ $behavior = "continuous" ]; then
    python src/train_baseline.py --eid $eid \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                $use_nlb \
                                --overwrite
elif [ $behavior = "choice" ] || [ $behavior = "block" ];
then
    python src/train_baseline.py --eid $eid \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior $behavior \
                                $use_nlb \
                                --overwrite
else
    echo "behavior: $behavior not supported"
fi

conda deactivate
cd script/yizi


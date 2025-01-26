#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train-rrr-multi"
#SBATCH --output="train-rrr-multi.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-48
#SBATCH --export=ALL

. ~/.bashrc

num_sessions=${1}
model_mode=${2}
behavior=${3}

conda activate ibl-mm

cd ../../

export WANDB_API_KEY="9e82a8134af3561339ab123ee336deec84d45b83"
export WANDB_DIR=/u/zwang34/multi_modal_foundation_model/cache

huggingface-cli login --token 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  --add-to-git-credential

if [ $behavior = "continuous" ]; then
    python src/train_baseline.py --num_sessions $num_sessions \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --overwrite
elif [ $behavior = "choice" ] || [ $behavior = "block" ];
then
    python src/train_baseline.py --num_sessions $num_sessions \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior $behavior \
                                --overwrite
elif [ $behavior = "all" ];
then
    python src/train_baseline.py --num_sessions $num_sessions \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior choice block wheel-speed whisker-motion-energy \
                                --overwrite
else
    echo "behavior: $behavior not supported"
fi

conda deactivate
cd script/yizi


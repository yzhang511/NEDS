#!/bin/bash

#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="eval-cpu"
#SBATCH --output="eval-cpu.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

num_sessions=${1}
eid=${2}
model_mode=${3}
mask_ratio=${4}
task_var=${5}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../..
if [ $model_mode = "mm" ]; then
    python src/eval_zero_shot.py --mask_mode temporal \
                                --mask_ratio ${mask_ratio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
                                --finetune \
                                --enc_task_var $task_var \
                                --overwrite	\
                                --data_path /scratch/bdtg/yzhang39/datasets/
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_zero_shot.py --mask_mode temporal \
                                --mask_ratio ${mask_ratio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
                                --enc_task_var $task_var \
                                --overwrite	\
                                --data_path /scratch/bdtg/yzhang39/datasets/
				--overwrite
else
    echo "model_mode: $model_mode not supported"
fi
cd script/yizi

conda deactivate
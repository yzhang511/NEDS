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
mask_rartio=${4}
task_var=${5}
search=${6}
use_nlb=${7}
nlb_bin_size=${8}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

if [ "$use_nlb" = "True" ]; then
    echo "Using NLB"
    use_nlb="--use_nlb"
else
    echo "Not using NLB"
    search=""
fi

# if search is empty, then we are not doing hyperparameter search
if [ "$search" = "True" ]; then
    echo "Doing hyperparameter search"
    search="--param_search"
else
    echo "Not doing hyperparameter search"
    search=""
fi

cd ../..
if [ $model_mode = "mm" ]; then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
            			        --overwrite \
                                --enc_task_var $task_var \
                                --data_path /projects/bcxj/yzhang39/datasets/ \
                                ${search} \
                                ${use_nlb} \
                                --nlb_bin_size $nlb_bin_size
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
				                --overwrite \
                                --enc_task_var $task_var \
                                --data_path /projects/bcxj/yzhang39/datasets/ \
                                ${search} \
                                ${use_nlb} \
                                --nlb_bin_size $nlb_bin_size
else
    echo "model_mode: $model_mode not supported"
fi
cd script/yizi

conda deactivate

#!/bin/bash
#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="eval"
#SBATCH --output="eval.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc

echo $TMPDIR

conda activate neds

cd ..

num_sessions=${1}
eid=${2}
model_mode=${3}
mask_rartio=${4}
task_var=${5}
search=${6}

user_name=$(whoami)
base_path="./" # change to your own path
data_path="/projects/bcxj/$user_name/datasets/"

if [ "$search" = "True" ]; then
    echo "Doing hyperparameter search"
    search="--param_search"
else
    echo "Not doing hyperparameter search"
    search=""
fi

if [ $model_mode = "mm" ]; then
    python src/eval_multi_modal.py --eid ${eid} \
                                --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --seed 42 \
                                --base_path $base_path \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
            			        --overwrite \
                                --enc_task_var $task_var \
                                --data_path $data_path \
                                ${search}
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_multi_modal.py --eid ${eid} \
                                --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --seed 42 \
                                --base_path $base_path \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
				                --overwrite \
                                --enc_task_var $task_var \
                                --data_path $data_path \
                                ${search}
else
    echo "model_mode: $model_mode not supported"
fi

conda deactivate

cd script

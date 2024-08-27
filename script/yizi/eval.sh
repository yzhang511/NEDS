#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train_eval"
#SBATCH --output="train_eval.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-5
#SBATCH --export=ALL

num_sessions=${1}
eid=${2}
model_mode=${3}
mask_rartio=${4}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../..
if [ $model_mode = "mm" ]; then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path /u/yzhang39/multi_modal_foundation_model \
                                --save_plot \
                                --mask_type embd \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb  
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path /u/yzhang39/multi_modal_foundation_model \
                                --save_plot \
                                --mask_type embd \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb 
else
    echo "model_mode: $model_mode not supported"
fi
cd script/ppwang

conda deactivate
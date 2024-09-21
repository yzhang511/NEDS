#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=eid_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00 
#SBATCH --mem=128g
#SBATCH --account=pr_136_general

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
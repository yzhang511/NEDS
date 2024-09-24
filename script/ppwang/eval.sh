#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=eval_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00 
#SBATCH --mem=128g
#SBATCH --account=pr_136_tandon_advanced

num_sessions=${1}
eid=${2}
model_mode=${3}
train_mode=${4} # finetune or train
mask_rartio=${5}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

finetune_arg=""
if [ $train_mode = "finetune" ]; then
    echo "finetune mode"
    finetune_arg="--finetune"
fi


cd ../..
if [ $model_mode = "mm" ]; then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --save_plot \
                                --mask_type embd \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                $finetune_arg \
                                --wandb  
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --save_plot \
                                --mask_type embd \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                $finetune_arg \
                                --wandb 
else
    echo "model_mode: $model_mode not supported"
fi
cd script/ppwang

conda deactivate
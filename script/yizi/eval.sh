#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="eval"
#SBATCH --output="eval.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-01
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
                                --base_path ./ \
                                --save_plot \
                                --mixed_training  \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
				--finetune \
			        --overwrite	
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/eval_multi_modal.py --mask_mode temporal \
                                --mask_ratio ${mask_rartio} \
                                --eid ${eid} \
                                --seed 42 \
                                --base_path ./ \
                                --save_plot \
                                --num_sessions ${num_sessions} \
                                --model_mode ${model_mode} \
                                --wandb \
				--overwrite
else
    echo "model_mode: $model_mode not supported"
fi
cd script/yizi

conda deactivate

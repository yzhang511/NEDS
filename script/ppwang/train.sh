#!/bin/bash

#SBATCH --job-name=train-mm
#SBATCH --output=train-mm-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
num_sessions=${1}
eid=${2}
model_mode=${3}
dummy_size=${4}
mask_ratio=${5}
echo $TMPDIR
conda activate ibl-mm

cd ../..

if [ $model_mode = "mm" ]; then
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --use_contrastive \
                                    --use_prompt \
                                    --use_moco \
                                    --dummy_load \
                                    --model_mode $model_mode 
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode 
else
    echo "model_mode: $model_mode not supported"
fi

cd script/ppwang

conda deactivate

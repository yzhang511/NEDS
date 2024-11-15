#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="mm"
#SBATCH --output="mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=2
#SBATCH -t 0-03
#SBATCH --export=ALL

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
    accelerate launch src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --multi_gpu
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    accelerate launch src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --multi_gpu
else
    echo "model_mode: $model_mode not supported"
fi

cd script/yizi

conda deactivate

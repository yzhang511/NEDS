#!/bin/bash

#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="mm-cpu"
#SBATCH --output="mm-cpu.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
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
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --data_path /scratch/bcxj/yzhang39/datasets/
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --data_path /scratch/bcxj/yzhang39/datasets/
else
    echo "model_mode: $model_mode not supported"
fi

cd script/yizi

conda deactivate

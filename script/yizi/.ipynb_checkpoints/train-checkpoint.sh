#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="mm"
#SBATCH --output="mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 2-00
#SBATCH --export=ALL

module load gpu
module load slurm

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

cd script/yizi

conda deactivate

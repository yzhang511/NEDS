#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --job-name="mm"
#SBATCH --output="mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-04:00:00
#SBATCH --export=ALL

. ~/.bashrc

num_sessions=${1}
eid=${2}
model_mode=${3}
dummy_size=${4}
mask_ratio=${5}
task_var=${6}
use_nlb=${7}
nlb_bin_size=${8}

echo $TMPDIR
conda activate ibl-mm

if [ "$use_nlb" = "True" ]; then
    echo "Using NLB"
    use_nlb="--use_nlb"
else
    echo "Not using NLB"
    search=""
fi

cd ../..

config_dir=$(pwd)/src/configs

if [ $model_mode = "mm" ]; then
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $use_nlb \
                                    --nlb_bin_size $nlb_bin_size \
                                    --data_path /projects/bcxj/yzhang39/datasets/ \
                                    --config_dir $config_dir
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $use_nlb \
                                    --nlb_bin_size $nlb_bin_size \
                                    --data_path /projects/bcxj/yzhang39/datasets/ \
                                    --config_dir $config_dir
else
    echo "model_mode: $model_mode not supported"
fi

cd script/yizi

conda deactivate

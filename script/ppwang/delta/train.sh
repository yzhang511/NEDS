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
#SBATCH --gpus=1
#SBATCH -t 0-05
#SBATCH --export=ALL

. ~/.bashrc

num_sessions=${1}
eid=${2}
model_mode=${3}
dummy_size=${4}
mask_ratio=${5}
task_var=${6}
search=${7}
echo $TMPDIR

conda activate ibl-mm

# if search is empty, then we are not doing hyperparameter search
if [ "$search" = "True" ]; then
    echo "Doing hyperparameter search"
    search="--search"
else
    echo "Not doing hyperparameter search"
    search=""
fi
cd ../../..
user_name=$(whoami)
if [ $model_mode = "mm" ]; then
    echo "Training multimodal model"
    config_dir=$(pwd)/src/configs
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $search \
                                    --config_dir $config_dir \
                                    --data_path /scratch/bdtg/yzhang39/datasets/
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    echo "Training $model_mode model"
    python src/train_multi_modal.py --eid $eid \
                                    --base_path ./ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $search \
                                    --data_path /scratch/bdtg/yzhang39/datasets/
else
    echo "model_mode: $model_mode not supported"
fi

cd script/ppwang/delta

conda deactivate

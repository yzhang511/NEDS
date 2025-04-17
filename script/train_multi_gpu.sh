#!/bin/bash

#SBATCH --account=beez-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --job-name="mm"
#SBATCH --output="mm.%j.out"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000   
#SBATCH -t 0-08:00:00
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

if [ "$use_nlb" = "True" ]; then
    echo "Using NLB"
    use_nlb="--use_nlb"
else
    echo "Not using NLB"
    search=""
fi

echo $TMPDIR

conda activate ibl-mm

cd ../..

config_dir=$(pwd)/src/configs

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export LAUNCHER="torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    "
export SCRIPT="src/train_multi_modal.py"

if [ $model_mode = "mm" ]; then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path ./ \
            --mask_ratio $mask_ratio \
            --mixed_training \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --enc_task_var $task_var \
            $use_nlb \
            --nlb_bin_size $nlb_bin_size \
            --data_path /projects/bcxj/yzhang39/datasets/ \
            --config_dir $config_dir
        "
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path ./ \
            --mask_ratio $mask_ratio \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --enc_task_var $task_var \
            $use_nlb \
            --nlb_bin_size $nlb_bin_size \
            --data_path /projects/bcxj/yzhang39/datasets/ \
            --config_dir $config_dir
        "
else
    echo "model_mode: $model_mode not supported"
fi

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun $CMD

cd script/yizi

conda deactivate

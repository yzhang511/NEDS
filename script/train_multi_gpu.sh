#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --job-name="train-multi-gpu"
#SBATCH --output="train-multi-gpu.%j.out"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000   
#SBATCH -t 0-04:00:00
#SBATCH --export=ALL

. ~/.bashrc

echo $TMPDIR

conda activate neds

cd ..

num_sessions=${1}
eid=${2}
model_mode=${3}
dummy_size=${4}
mask_ratio=${5}
task_var=${6}

user_name="yzhang39"
base_path="./" # change to your own path
config_dir=$(pwd)/src/configs
data_path="/projects/bcxj/$user_name/datasets/"

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
export SCRIPT="src/train.py"

if [ $model_mode = "mm" ]; then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path $base_path \
            --mask_ratio $mask_ratio \
            --mixed_training \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --enc_task_var $task_var \
            --data_path $data_path \
            --config_dir $config_dir
        "
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path $base_path \
            --mask_ratio $mask_ratio \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --enc_task_var $task_var \
            --data_path $data_path \
            --config_dir $config_dir
        "
else
    echo "model_mode: $model_mode not supported"
fi

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 

srun $CMD

conda deactivate

cd script

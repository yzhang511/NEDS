#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="pretrain"
#SBATCH --output="pretrain.%j.out"
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000   
#SBATCH -t 2-00
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
export SCRIPT="src/train_multi_modal_tmp.py"

if [ $model_mode = "mm" ]; then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path /scratch/bdtg/yzhang39/tmp/ \
            --mask_ratio $mask_ratio \
            --mixed_training \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --data_path /scratch/bdtg/yzhang39/datasets/
        "
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    export SCRIPT_ARGS=" \
            --eid $eid \
            --base_path /scratch/bdtg/yzhang39/tmp/ \
            --mask_ratio $mask_ratio \
            --num_sessions $num_sessions \
            --dummy_size $dummy_size \
            --model_mode $model_mode \
            --multi_gpu \
            --data_path /scratch/bdtg/yzhang39/datasets/
        "
else
    echo "model_mode: $model_mode not supported"
fi

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun $CMD

cd script/yizi

conda deactivate

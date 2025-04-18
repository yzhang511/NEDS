#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000
#SBATCH -t 0-06
#SBATCH --export=ALL

set -x

. ~/.bashrc

echo $TMPDIR

conda activate neds

cd ..

num_sessions=${1}
eid=${2}
train_mode=${3}
model_mode=${4}
dummy_size=${5}
mask_ratio=${6}
search=${7}
task_var=${8}

user_name=$(whoami)
config_dir=$(pwd)/src/configs
data_path="/projects/bcxj/$user_name/datasets/"

if [ "$search" = "True" ]; then
    echo "Doing hyperparameter search"
    search="--search"
    base_path="/projects/beez/$user_name/tune/session_$num_sessions/"
else
    echo "Not doing hyperparameter search"
    search=""
    base_path="./" # change to your own path
fi

if [ $train_mode = "finetune" ]; then
    python_file="src/finetune.py"
else
    python_file="src/train.py"
fi

if [ "$search" = "--search" ]; then
    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # If we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
    else
    head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    # Starting the Ray head node
    port=1111
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "1" --num-gpus "1" --block &

    # Starting the Ray worker nodes
    sleep 10

    worker_num=$((SLURM_JOB_NUM_NODES - 1))  # number of nodes other than the head node

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "1" --num-gpus "1" --block &
        sleep 5
    done
fi

if [ $model_mode = "mm" ]; then
    echo "Training multimodal model:"
    python $python_file --eid $eid \
    --base_path $base_path \
    --mask_ratio $mask_ratio \
    --mixed_training \
    --num_sessions $num_sessions \
    --dummy_size $dummy_size \
    --model_mode $model_mode \
    --enc_task_var $task_var \
    $search \
    --num_tune_sample 30 \
    --config_dir $config_dir \
    --data_path $data_path
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    echo "Training $model_mode model:"
    python $python_file --eid $eid \
    --base_path $base_path \
    --mask_ratio $mask_ratio \
    --num_sessions $num_sessions \
    --dummy_size $dummy_size \
    --model_mode $model_mode \
    --enc_task_var all \
    $search \
    --num_tune_sample 30 \
    --config_dir $config_dir \
    --data_path $data_path
else
    echo "model_mode: $model_mode not supported"
fi

conda deactivate

cd script

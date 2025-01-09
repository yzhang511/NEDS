#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="tune"
#SBATCH --output="tune.%j.out"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000
#SBATCH -t 0-04
#SBATCH --export=ALL

set -x

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

cd ../..
user_name=$(whoami)

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
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

if [ $model_mode = "mm" ]; then
    echo "Training multimodal model:"
    config_dir=$(pwd)/src/configs
    python src/train_multi_modal.py --eid $eid \
                                    --base_path /scratch/bdtg/$user_name/tune/ \
                                    --mask_ratio $mask_ratio \
                                    --mixed_training \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $search \
                                    --num_tune_sample 40 \
                                    --config_dir $config_dir \
                                    --data_path /scratch/bdtg/$user_name/datasets/
elif [ $model_mode = "encoding" ] || [ $model_mode = "decoding" ];
then
    echo "Training $model_mode model:"
    python src/train_multi_modal.py --eid $eid \
                                    --base_path /scratch/bdtg/$user_name/tune/ \
                                    --mask_ratio $mask_ratio \
                                    --num_sessions $num_sessions \
                                    --dummy_size $dummy_size \
                                    --model_mode $model_mode \
                                    --enc_task_var $task_var \
                                    $search \
                                    --num_tune_sample 40 \
                                    --config_dir $config_dir \
                                    --data_path /scratch/bdtg/$user_name/datasets/
else
    echo "model_mode: $model_mode not supported"
fi


cd script/ppwang/delta

conda deactivate

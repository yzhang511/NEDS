#!/bin/bash

#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="create_dataset"
#SBATCH --output="create_dataset.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL


. ~/.bashrc
eid=${1}
model_mode=mm
dummy_size=0
mask_ratio=0.1
echo $TMPDIR
conda activate ibl-mm

# get user name
user_name=$(whoami)
cd ../../..
python src/create_dataset.py --eid $eid \
                                --base_path ./\
                                --mask_ratio $mask_ratio \
                                --mixed_training \
                                --num_sessions 1 \
                                --dummy_size $dummy_size \
                                --model_mode $model_mode \
                                --data_path /scratch/bdtg/${user_name}/datasets/

cd script/ppwang/delta


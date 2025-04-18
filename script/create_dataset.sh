#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc

num_sessions=${1}
eid=${2}

echo $TMPDIR

conda activate neds

cd ..

user_name=$(whoami)

python src/create_dataset.py --eid $eid \
                             --num_sessions $num_sessions \
                             --model_mode mm \
                             --mask_ratio 0.1 \
                             --mixed_training \
                             --base_path ./ \
                             --data_path /projects/bcxj/$user_name/datasets/

conda deactivate

cd script

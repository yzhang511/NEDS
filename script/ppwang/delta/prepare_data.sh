#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="prep_data"
#SBATCH --output="prep_data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH --mem 100000
#SBATCH -t 0-04
#SBATCH --export=ALL
. ~/.bashrc
echo $TMPDIR
user_name=$(whoami)
cd ../../..
conda activate ibl-mm
python src/prepare_data.py --base_path /scratch/bdtg/${user_name}/datasets --n_sessions 200
cd script/ppwang/delta
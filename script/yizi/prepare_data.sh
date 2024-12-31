#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH --mem 100000
#SBATCH -t 0-04
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR

cd ../..
conda activate ibl-mm
python src/prepare_data.py --base_path /scratch/bdtg/yzhang39/datasets --n_sessions 200


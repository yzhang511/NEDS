#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH --mem 150000
#SBATCH -t 2-00
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR

cd ../..
conda activate ibl-mm
python src/prepare_data.py --base_path /scratch/bcxj/yzhang39/Downloads --n_sessions 200


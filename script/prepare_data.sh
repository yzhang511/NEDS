#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 100000
#SBATCH -t 0-02
#SBATCH --export=ALL

. ~/.bashrc

echo $TMPDIR

cd ..

conda activate neds

python src/prepare_data.py --base_path /projects/bcxj/yzhang39/datasets --n_sessions 100

conda deactivate

cd script

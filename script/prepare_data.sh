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

num_sessions=${1:-1}  # Default to 1 if not provided
eid=${2:-"None"}      # Default to "None" if not provided

if ! [[ "$num_sessions" =~ ^[0-9]+$ ]]; then
    echo "Error: num_sessions must be an integer"
    exit 1
fi

if [ "$num_sessions" -eq 1 ]; then
    echo "Download data for single session"
    if [ "$eid" = "None" ]; then
        echo "Error: eid must be provided for single session"
        exit 1
    fi
else
    echo "Download data for multiple sessions"
    eid="None"
fi

echo $TMPDIR

cd ..

conda activate neds

user_name=$(whoami)

python src/prepare_data.py --n_sessions $num_sessions \
                           --eid $eid \
                           --base_path /projects/bcxj/$user_name/datasets

conda deactivate

cd script

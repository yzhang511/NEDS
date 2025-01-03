#!/bin/bash
  
#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="eval-rrr"
#SBATCH --output="eval-rrr.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR

eid=${1}
model_mode=${2}
behavior=${3}

conda activate ibl-mm

cd ../..

if [ $behavior = "continuous" ]; then
    python src/eval_baseline.py --eid $eid \
                                --seed 42 \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --save_plot \
                                --overwrite \
                                --wandb
elif [ $behavior = "choice" ] || [ $behavior = "block" ];
then
    python src/eval_baseline.py --eid $eid \
                                --seed 42 \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior $behavior \
                                --save_plot \
                                --overwrite \
                                --wandb
else
    echo "behavior: $behavior not supported"
fi

conda deactivate
cd script/yizi

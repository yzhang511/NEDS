#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="eval_mm"
#SBATCH --output="eval_mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-1
#SBATCH --export=ALL


. ~/.bashrc
echo $TMPDIR

conda activate ibl-mm

cd ../

python src/eval_multi_modal.py --mask_mode temporal \
                               --mask_ratio 0.3 \
                               --eid db4df448-e449-4a6f-a0e7-288711e7a75a \
                               --wandb \
                               --seed 42 \
                               --base_path /scratch/bcxj/yzhang39 \
                               --save_plot \
                               --mask_type embd \
                               --use_MtM

conda deactivate


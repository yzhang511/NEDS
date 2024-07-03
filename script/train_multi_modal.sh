#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train_mm"
#SBATCH --output="train_mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-8
#SBATCH --export=ALL


. ~/.bashrc
echo $TMPDIR

conda activate ibl-mm

cd ../

python src/train_multi_modal.py --mask_ratio 0.3 \
                                --eid 51e53aff-1d5d-4182-a684-aba783d50ae5 \
                                --base_path /scratch/bcxj/yzhang39 \

# check mask_type in train_mm.yaml
# --use_MtM
# check the mask_mode in train_mm.yaml
# --mask_mode temporal 

conda deactivate


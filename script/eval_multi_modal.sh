#!/bin/bash
#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="train_mm"
#SBATCH --output="train_mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-1
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
echo $TMPDIR

conda activate ibl-mm

cd /home/yzhang39/multi_modal_foundation_model

huggingface-cli login --token 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  --add-to-git-credential

python src/eval_multi_modal.py --mask_mode temporal --mask_ratio 0.3 --train --eid 51e53aff-1d5d-4182-a684-aba783d50ae5

conda deactivate


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
pretrain_n_ses=${4}

conda activate ibl-mm

cd ../..

export WANDB_API_KEY="9e82a8134af3561339ab123ee336deec84d45b83"
export WANDB_DIR=/u/zwang34/multi_modal_foundation_model/cache

huggingface-cli login --token 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  --add-to-git-credential

if [ $behavior = "continuous" ]; then
    python src/eval_baseline.py --eid $eid \
                                --seed 42 \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --save_plot \
                                --overwrite \
                                --finetune \
                                --pretrain_num_sessions $pretrain_n_ses \
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
                                --finetune \
                                --pretrain_num_sessions $pretrain_n_ses \
                                --wandb
elif [ $behavior = "all" ];
then
    python src/eval_baseline.py --eid $eid \
                                --seed 42 \
                                --base_path ./ \
                                --data_path /scratch/bdtg/yzhang39/datasets/ \
                                --model_mode $model_mode \
                                --behavior choice block wheel-speed whisker-motion-energy \
                                --save_plot \
                                --overwrite \
                                --finetune \
                                --pretrain_num_sessions $pretrain_n_ses \
                                --wandb
else
    echo "behavior: $behavior not supported"
fi

conda deactivate
cd script/yizi

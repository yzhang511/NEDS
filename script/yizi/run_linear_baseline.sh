#!/bin/bash
#SBATCH --account=beez-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="linear"
#SBATCH --output="linear.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

eid=${1}
model_mode=${2}
model=${3}
use_nlb=${4}
nlb_bin_size=${5}

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

if [ "$use_nlb" = "True" ]; then
    echo "Using NLB"
    use_nlb="--use_nlb"
else
    echo "Not using NLB"
    search=""
fi

cd ../..
if [ $model_mode = "encoding" ]; then
    python src/run_linear_encoder.py \
                                --eid ${eid} \
                                --base_path ./ \
                                --model ${model} \
                                --data_path /projects/bcxj/yzhang39/datasets/ \
                                ${use_nlb} \
                                --nlb_bin_size ${nlb_bin_size}
elif [ $model_mode = "decoding" ];
then
    python src/run_linear_decoder.py \
                                --eid ${eid} \
                                --base_path ./ \
                                --data_path /projects/bcxj/yzhang39/datasets/ \
                                ${use_nlb} \
                                --nlb_bin_size ${nlb_bin_size}
else
    echo "model: $model not supported"
fi
cd script/yizi

conda deactivate

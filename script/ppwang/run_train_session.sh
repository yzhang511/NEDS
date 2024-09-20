#!/bin/bash

num_sessions=${1}
model_mode=${2} # mm, decoding, encoding

if [ $num_sessions -eq 1 ]
then
    while IFS= read -r line
    do
        echo "Train on ses eid: $line"
        sbatch --gres=gpu:rtx8000:1 -t 18:00:00  train.sh 1 $line $model_mode 80000 0.1
    done < "../../data/test_eids.txt"
fi

if [ $num_sessions -gt 1 ]
then
    echo "Train on multi-session"
    sbatch --gres=gpu:rtx8000:1 -t 7-00:00:00  train.sh $num_sessions none $model_mode 80000 0.1
fi
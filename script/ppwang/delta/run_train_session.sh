#!/bin/bash

num_sessions=${1}
model_mode=${2} # mm, decoding, encoding
train_mode=${3} # train, finetune, search
# Check if finetune mode requires more than 1 session
if [ "$train_mode" == "finetune" ] && [ "$num_sessions" -le 1 ]; then
    echo "Error: Finetuning requires more than one session."
    exit 1
fi
if [ "$train_mode" == "search" ]; then
    search=True
else
    search=False
fi

if [ $num_sessions -eq 1 ]
then
    while IFS= read -r line
    do
        if [ $train_mode == "train" ] || [ $train_mode == "search" ]
        then
            echo "Train on ses eid: $line"
            sbatch  train.sh 1 $line $model_mode 0 0.1 all $search
        fi
    done < "../../../data/test_eids.txt"
fi

if [ $num_sessions -gt 1 ]
then
    if [ $train_mode == "train" ] || [ $train_mode == "search" ]
    then
        echo "Pretrain on multi-session"
        sbatch train.sh $num_sessions none $model_mode 0 0.1 all $search
    fi
    while IFS= read -r line
    do
        if [ $train_mode == "finetune" ]
            then
                echo "Finetune on ses eid: $line, with pretrain model on $num_sessions sessions"
                sbatch finetune.sh $num_sessions $line $model_mode 0 0.1
        fi
    done < "../../../data/test_eids.txt"
fi
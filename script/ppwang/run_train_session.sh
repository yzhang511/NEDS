#!/bin/bash

num_sessions=${1}
# if num_sessions is 1, run each session separately
# elif multi-session, give session number as argument

if [ $num_sessions -eq 1 ]
then
    while IFS= read -r line
    do
        echo "Train on ses eid: $line"
        sbatch --gres=gpu:1 -t 12:00:00  train.sh 1 $line 0.1 50000
    done < "../../data/train_eids.txt"
fi

if [ $num_sessions -gt 1 ]
then
    echo "Train on multi-session"
    sbatch --gres=gpu:h100:1 -t 3-12:00:00  train.sh $num_sessions none 0.1 60000
fi
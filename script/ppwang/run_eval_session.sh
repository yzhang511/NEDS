#!/bin/bash

num_sessions=${1}
while IFS= read -r line
do
    echo "Eval on ses eid: $line"
    sbatch eval.sh $num_sessions $line 0.1
done < "../../data/train_eids.txt"
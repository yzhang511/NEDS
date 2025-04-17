#!/bin/bash

num_sessions=${1}
model_mode=${2} # mm, decoding, encoding
task_var=${3}
while IFS= read -r line
do
    echo "Eval on ses eid: $line"
    sbatch eval_cpu.sh $num_sessions $line $model_mode 0.1
done < "../../data/train_eids.txt"
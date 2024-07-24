#!/bin/bash

while IFS= read -r line
do
    echo "Train on ses eid: $line"
    sbatch train.sh 1 $line 0.1
done < "../../data/train_eids.txt"
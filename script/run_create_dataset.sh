#!/bin/bash

while IFS= read -r line
do
    echo "Create dataset on ses eid: $line"
    sbatch create_dataset.sh 1 $line

done < "../data/test_eids.txt"

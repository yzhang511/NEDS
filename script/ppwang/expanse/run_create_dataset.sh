#!/bin/bash

while IFS= read -r line
do
    echo "Create dataset on ses eid: $line"
    sbatch create_dataset.sh $line
done < "../../../data/eids.txt"
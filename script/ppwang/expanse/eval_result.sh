#!/bin/bash

num_sessions=${1}
model_mode=${2} # mm, decoding, encoding

cd ../../..
conda activate ibl-mm

python src/eval_result.py --log_dir results \
                          --model_mode $model_mode \
                          --num_sessions $num_sessions \
                          --mixed_training 


conda deactivate

cd script/ppwang/expanse
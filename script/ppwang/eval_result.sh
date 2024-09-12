#!/bin/bash

num_sessions=${1}
model_mode=${2} # mm, decoding, encoding

cd ../..
conda activate ibl-mm

python src/eval_result.py --log_dir results \
                          --model_mode $model_mode \
                          --use_contrastive \
                          --mixed_training \
                          --use_prompt \
                          --use_moco \
                          --num_sessions $num_sessions \
                          --eval_session_path data/train_eids.txt
                        #   --eval_session_path data/filtered_eids.txt


conda deactivate

cd script/ppwang
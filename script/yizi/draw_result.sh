#!/bin/bash

. ~/.bashrc
conda activate ibl-mm

cd ../..

python src/draw/draw_result.py --result_dir ./results/

cd script/ppwang
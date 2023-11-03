#!/bin/bash
python test.py \
    --test_file ${1} \
    --model_name_or_path ./output \
    --batch_size 2 \
    --num_beams 5 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_path ${2}
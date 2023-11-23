#!/bin/bash
python test.py \
    --model_name_or_path ${1} \
    --peft_model ${2} \
    --test_file ${3} \
    --output_path ${4} \
    --batch_size 2

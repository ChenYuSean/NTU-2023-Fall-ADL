## Download and Run
Download the model by running
```bash
bash download.sh
```
Then run below for predictions
```bash
bash run.sh /path/to/base_model path/to/peft_model /path/to/input.json /path/to/output.json
```

## Run ppl.py
```bash
python ppl.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --peft_path output/checkpoint-250 \
    --test_data_path data/public_test.json
```

## Reproduce Training
```bash
python train.py \
    --train_file data/train.json \
    --validation_file data/public_test.json \
    --model_name_or_path Taiwan-LLM-7B-v2.0-chat \
    --train_size 1000 \
    --max_step 300 \
    --checkpointing_steps 50 \
    --learning_rate 2e-4 \
    --output_dir output
```
**Arguments**
- train_file: path to train file
- validation_file: path to validation file
- model_name_or_path: path to base model
- train_size: the number of sample in train data
- max_step: total train step

## Reproduce Testing without run.sh
```bash
python test.py \
    --test_file data/private_test.json \
    --model_name_or_path Taiwan-LLM-7B-v2.0-chat \
    --peft_model output/checkpoint-250 \
    --batch_size 2 \
    --output_path ./prediction.json
```



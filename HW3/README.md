## Download and Run
Download the model by running
```bash
bash download.sh
```
Then run below for predictions
```bash
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
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


## Reproduce Testing without run.sh
```bash
python test.py \

```
**Arguments**

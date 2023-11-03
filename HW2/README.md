## Download and Run
Download the model by running
```bash
bash download.sh
```
Then run below for predictions
```
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
```
## Reproduce Training
```bash
python train_t5.py \
    --train_file ./data/train.jsonl \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 5 \
    --num_train_epochs 5 \
    --output_dir ./output 
```
**Arguments**
- train_file: path to train file
- batch_size: batch size, default = 1
- gradient_accumulation_steps:  gradient accumulation steps,  
            default = 1
- learning_rate: learning rate, default = 5e-5
- num_train_epoch: # of epochs during training
- max_source_length: maximum length of input
- max_target_length: maximum length of valid prediction
- num_beams: # of beams used in valid
- output_dir: path to output directory 

## Reproduce Testing without run.sh
```bash
python test.py \
    --test_file ./data/public.jsonl \
    --model_name_or_path ./output \
    --batch_size 2 \
    --num_beams 5 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_path ./submission.jsonl
```
**Arguments**
- test_file: path to test file
- batch_size: batch size used in testing
- max_source_length: maximum length of input
- max_target_length: maximum length of prediction
- output_path: path to output jsonl

**Strategy Args**  
Toggle on by adding the flags
- No Flags: Greedy as default
- num_beams: # of beams in beam search 
- top_k: first k in top-k 
- top_p: probability p in top-p
- temperature: t in temperature
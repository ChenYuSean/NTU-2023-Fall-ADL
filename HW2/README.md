```bash
python train_t5.py 
    --train_file ./data/train.jsonl
    --batch_size 1
    --gradient_accumulation_steps 2
    --source_prefix ""
    --max_source_length 1024
    --max_target_length 128
    --num_beams 5
    --num_train_epochs 5
    --output_dir ./output 
```
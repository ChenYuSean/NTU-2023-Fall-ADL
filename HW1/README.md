## Training Paragraph Selection Model
```bash
python ParagraphSelection.py  \
    --train_file ./train.json\
    --validation_file ./valid.json \
    --context_file ./context.json \
    --max_seq_length 512 \
    --model_name_or_path bert-base-chinese \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --output_dir ./output_PS

```

# Training Question Answering Model
```bash
python QuestionAnswering.py \
    --train_file ./train.json \
    --validation_file ./valid.json \
    --context_file ./context.json \
    --max_seq_length 512 \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --output_dir ./output_QS
```


# Predict

```bash
python predict.py \
    --test_file ./test.json \
    --context_file ./context.json \
    --per_device_batch_size 8 \
    --max_seq_length 512 \
    --n_best_size 20 \
    --doc_stride 32 \
    --output_path ./prediction.csv
```
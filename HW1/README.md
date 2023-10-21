```bash
python predict.py   --test_file ./test.json \
                    --per_device_batch_size 8 \
                    --max_seq_length 512 \
                    --n_best_size 20 \
                    --doc_stride 32 \
                    --output_path ./prediction.csv
```
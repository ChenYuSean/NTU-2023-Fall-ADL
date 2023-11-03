# %%
#!/usr/bin/env python
# coding=utf-8
# import ipdb

import argparse
import os
import torch
import nltk
import numpy as np
import json
import jsonlines

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

# %%
# Global variables
USE_NOTEBOOK_LAUNCHER = False
MODEL_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
RESULTs = None
str_args = None

# %%
# # Comment out when using .py file
# str_args = [
#     "--test_file", "./data/public_subset.jsonl",
#     "--model_name_or_path", "./output",
#     "--batch_size", "2",
#     "--num_beams", "5",
#     # "--top_k", "10",
#     #"--top_p", "0.9",
#     # "--temperature", "1.0",
#     "--max_source_length", "1024", 
#     "--max_target_length", "128",
#     "--output_path", "./OUTPUTS/Predictions/Beam_5.jsonl" 
# ]

# %%
def parse_args(str_args = None):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--test_file", type=str ,required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./prediction.jsonl"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default = "google/mt5-small"
    )
    # Predicting Parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )
    # Preprocessing
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
    )    
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
    )

    
    args = parser.parse_args(str_args)
    return args

# %%
def main(str_args = None):
    args = parse_args(str_args)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Prepare 
    if args.seed is not None:
        set_seed(args.seed)
        
    if accelerator.is_main_process: 
        if args.output_path is not None:
            os.makedirs(os.path.join(*args.output_path.split("/")[:-1]), exist_ok=True)
    accelerator.wait_for_everyone()
        
    # Load Dataset
    data_files ={}
    data_files['test'] = args.test_file
    raw_datasets = load_dataset("json", data_files=data_files)
    
    # Load Model
    config = AutoConfig.from_pretrained(os.path.join(args.model_name_or_path,CONFIG_NAME))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join(args.model_name_or_path,MODEL_NAME),
            config=config
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    prefix = args.source_prefix if args.source_prefix is not None else ""
    # Preprocessing the datasets.
    # First we tokenize all the texts.    
    column_names = raw_datasets["test"].column_names
    text_column = 'maintext'
    
    max_target_length = args.max_target_length
    padding = False
    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs
    
    with accelerator.main_process_first():
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
        )

    # Data Collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of = 8 if accelerator.use_fp16 else None,
    )

    # Postprocessing the predictions
    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds
    
    # Data Loader
    
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    
    # Prepare everything with our `accelerator`.
    model,test_dataloader = accelerator.prepare(
        model,test_dataloader
    )
    
    # Evaluation
    model.eval()
    preds = []
    do_sample = True if args.top_k is not None or args.top_p is not None else False
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
        "top_k" : args.top_k,
        "top_p" : args.top_p,
        "temperature" : args.temperature,
        "do_sample": do_sample
    }
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)
            preds += decoded_preds
    
    if args.output_path is not None:
        accelerator.wait_for_everyone()
        # Save predictions
        results = list(map(lambda pred,id: {'title': pred, 'id': id}, preds, raw_datasets['test']['id']))
        with jsonlines.open(args.output_path,'w') as writer:
            for result in results:
                writer.write(result)


# %%
if __name__ == "__main__":
    if USE_NOTEBOOK_LAUNCHER:
        notebook_launcher(main,(str_args,), num_processes=1)
    else:      
        main(str_args)



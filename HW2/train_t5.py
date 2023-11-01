# %%
#!/usr/bin/env python
# coding=utf-8

# debug
# import ipdb
#
import argparse
import os
import random
import math
import datasets
import evaluate
import torch
import nltk
import numpy as np
import csv
import json

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

# %%
# Global variables
USE_NOTEBOOK_LAUNCHER = False
str_args = None

# %%
# Comment out when using .py file
# str_args = [
#     "--train_file", "./data/train.jsonl",
#     "--batch_size", "1",
#     "--gradient_accumulation_steps", "2",
#     "--source_prefix", "",
#     "--max_source_length", "1024", 
#     "--max_target_length", "128",
#     "--num_beams", "5",
#     "--num_train_epochs", "5",
#     "--checkpointing_steps", "2500",
#     "--output_dir", "./output"    
# ]

# %%
def parse_args(str_args = None):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_file", type=str ,required=True)
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output"
    )
    parser.add_argument(
        "--split_ratio",
        type = float,
        default= 0.1
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default = "google/mt5-small"
    )
    # Training Parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
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
        default=1,
    )
    
    # Checkpoint
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    
    args = parser.parse_args(str_args)
    return args

# %%
def main(str_args = None):
    args = parse_args(str_args)
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision = None
    )
    
    # Prepare 
    if args.seed is not None:
        set_seed(args.seed)
        
    if accelerator.is_main_process: 
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
        
    # Load Dataset
    split = load_dataset("json", data_files=args.train_file,split='train').train_test_split(test_size=args.split_ratio)
    raw_datasets = DatasetDict({'train': split['train'], 'validation': split['test']})
    del split
    # Load Model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
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
    column_names = raw_datasets["train"].column_names
    text_column = 'maintext'
    summary_column = 'title'
    
    max_target_length = args.max_target_length
    padding = False
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
        )
        eval_dataset = raw_datasets["validation"].map(
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
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    # Data Loader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
        
    # Metric
    # Use the get_rouge frome tw_rouge
    # metric = evaluate.load("rouge")

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    all_results = []
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        # Training
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # Evaluation
        model.eval()
        preds = []
        refs = []
        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                preds += decoded_preds
                refs += decoded_labels
                
        result = {"pred":preds,"ref":refs}
        all_results.append(result)
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            
            # Save the prediction
            with open(os.path.join(args.output_dir,"valid_result.json"), "w") as f:
                f.write(json.dumps(all_results,indent = 4))

# %%
if __name__ == "__main__":
    if USE_NOTEBOOK_LAUNCHER:
        notebook_launcher(main,(str_args,), num_processes=2)
    else:      
        main(str_args)



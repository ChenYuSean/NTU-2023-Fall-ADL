# %%
# !pip install torch==2.1.0
# !pip install transformers==4.34.1
# !pip install bitsandbytes
# !pip install peft==0.6.0
# !pip install datasets
# !pip install evaluate
# !pip install accelerate
# !pip install sentencepiece
# !pip install einops
# !pip install scikit-learn
# !pip install ipdb

# %%
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from accelerate import notebook_launcher
from accelerate import Accelerator
from utils import get_bnb_config
from utils import get_prompt
from torch.utils.data import DataLoader

# %%
# Global variables
ROOT_PATH = './'
str_args = None

# %%
# Parser
def parse_args(str_args = None):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_file", type=str ,required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="./prediction.json"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default = None
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default = None
    )
    # Trainer Parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--source_max_len",
        type=int,
        default=1024,
    )
    # Generation Argument
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None
    )
    parser.add_argument(
        "--do_sample",
        action='store_true'
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1
    )

    args = parser.parse_args(str_args)
    return args

# %%
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        IGNORE_INDEX = -100
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input for causal LM
        input_ids = []
        for tokenized_source in tokenized_sources_with_prompt['input_ids']:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict

# %%
def main(str_args = None):
    args = parse_args(str_args)

    # Prepare
    logger = logging.getLogger(__name__)
    compute_dtype = torch.float16
    if args.seed is not None:
        set_seed(args.seed)
    if args.output_path is not None:
        output_dir = os.path.join(*args.output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print('Load Dataset')
    def format_dataset(dataset):
        def processing(example):
            return {'input': get_prompt(example['instruction'])}
        dataset = dataset.map(processing)
        # Remove 
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col in ['instruction','output']]
        )
        return dataset

    raw_dataset = load_dataset("json", data_files=args.test_file,split='train')
    test_dataset = format_dataset(raw_dataset)

    # Load Model
    print('Load Model')
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config = bnb_config,
        load_in_4bit = True,
        torch_dtype=compute_dtype,
        device_map = 'cuda:0'
    )
    base_model.config.torch_dtype=compute_dtype
    # Load PeftModel
    print("Loading adapters.")
    model = PeftModel.from_pretrained(base_model, args.peft_model)

    # Load Tokenizer
    print('Load Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        tokenizer_type='llama'
    )
    tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(base_model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(base_model.config.bos_token_id),
        "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
    })
    # Data Collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len
    )
    # Generatrion Config
    gen_config = transformers.GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        min_new_tokens = args.min_new_tokens,
        do_sample = args.do_sample,
        num_beams = args.num_beams,
        num_beam_groups = args.num_beam_groups,
        )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    progress = tqdm(total=len(test_dataloader))
    model.eval()
    all_predictions=[]
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            predictions = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config = gen_config,
            )
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            all_predictions += predictions
            progress.update()

    with open(args.output_path, 'w') as fout:
        outputs = []
        for i, example in enumerate(test_dataset):
            output_example = {}
            output_example['id'] = example['id'] 
            output_example['output'] = all_predictions[i].replace(example['input'], '').strip()
            outputs.append(output_example)
        fout.write(json.dumps(outputs,indent=4,ensure_ascii=False))

# %%
if __name__ == "__main__":
    main()
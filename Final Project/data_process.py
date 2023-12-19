# %%
import copy
import json
import os
import sys
import numpy as np
import random

import datasets
from datasets import Dataset
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse


# %%
def read_json_files(path, max_file_num:int = None, max_sample_per_file: int = None, shuffle = True):
    json_files_content = {}
    content_list = []
    # Get the file names
    files = []
    for root, dirs, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.json') and file != "star_record.json":
                files.append(os.path.join(root, file))  
    # Shuffle the list
    if shuffle:
        random.shuffle(files)
    # Read the files
    file_num = 0         
    for file in files:
        file_num = file_num + 1
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if shuffle:
                random.shuffle(content)
            # json_files_content[file] = content
            for record in content:
                if type(record['votes']) == str:
                    if '萬' in record['votes']:
                        record['votes'] = int(record['votes'].replace('萬', '')) * 10000
            if max_sample_per_file is not None:
                content_list += content[:max_sample_per_file]
            else:
                content_list += content
        if max_file_num is not None and file_num >= max_file_num:
            break
    return content_list


# %%
def get_prompt(title:str, description:str, star_num:str, mood:str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    
    seed = random.random()
    comment_type = '好評' if star_num.split('.')[0] in ['4', '5'] else '差評' if star_num.split('.')[0] in ['1', '2'] else '中立評論'
    if mood == 'like':
        mood = '喜歡'
    elif mood == 'happiness':
        mood = '開心'
    elif mood == 'sadness':
        mood = '難過'
    elif mood == 'disgust':
        mood = '厭惡'
    elif mood == 'anger':
        mood = '生氣'
    elif mood == 'surprise':
        mood = '驚訝'
    elif mood == 'fear':
        mood = '害怕'
    elif mood == 'none':
        mood = '中立'
    
    return f"{seed} 你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、詳細的回答。\
USER: 請幫這部影片生出對應需求的{comment_type}。影片標題:[{title}]。影片敘述:[{description}]。需求情感:[{mood}]。\
ASSISTANT:"


# %%
if __name__ == '__main__':
    PATH = "./train_data/"
    content = read_json_files(PATH, 1, None, True)
    print(content[:5])
    dataset = Dataset.from_list(content)
    print(dataset)




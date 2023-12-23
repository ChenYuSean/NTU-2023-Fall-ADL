# %%
import copy
import json
import os
import sys
import numpy as np
import random
import pandas as pd

import datasets
from datasets import Dataset
import torch
import transformers
import argparse

import re

# %%
def remove_emoji_in_string(input_string):
    emoji_pattern = re.compile("["
                            u"\U00002700-\U000027BF"  # Dingbats
                            u"\U0001F600-\U0001F64F"  # Emoticons
                            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
                            u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
                            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                            u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', input_string)

# %%
def read_json_files(path, remove_emoji = False) -> pd.DataFrame:
    '''
    Returns a dataframe of root directory, file names, and content(list) of json files
    '''
    def preprocessing(content, remove_emoji):
        content = copy.deepcopy(content)
        for record in content:
            # character to str
            if type(record['votes']) == str:
                if '萬' in record['votes']:
                    record['votes'] = int(record['votes'].replace('萬', '')) * 10000
            # emoji remove
            if remove_emoji:
                record['video_title'] = remove_emoji_in_string(record['video_title'])
                record['video_description'] = remove_emoji_in_string(record['video_description'])
                record['comment_text'] = remove_emoji_in_string(record['comment_text'])
        return content

    roots = []
    file_names = []
    contents = []
    # Fast return if the path is a file
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        content = preprocessing(content,remove_emoji)
        return pd.DataFrame({'root': [path], 'file_name': [path.split('/')[-1]], 'content': [content]})
    
    # Get the root, file names, and content
    for root, dirnames, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.json') and file != "star_record.json":
                with open(os.path.join(root,file), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                content = preprocessing(content,remove_emoji)
                # create dataframe
                roots.append(root)
                file_names.append(file)
                contents.append(content)
    return pd.DataFrame({'root': roots, 'file_name': file_names, 'content': contents})


# %%
def data_selection(file_info: pd.DataFrame, num_video_per_channel = None, select = True, seed = None) -> pd.DataFrame :
    '''
    Randomly select data from each file and return a new dataframe of training data
    num_video_per_channel: number of video to select from each channel. If None, select all videos
    '''
    random.seed(seed)
    moods = ['like','happiness','sadness','anger','fear','surprise','disgust']
    
    datalist = []
    channels = file_info['root'].unique()
    for channel in channels:
        videos = file_info.loc[file_info['root'] == channel, ['file_name','content']]
        if num_video_per_channel is not None:
            videos = videos.sample(n = num_video_per_channel, random_state = seed).reset_index(drop=True)
        for vid in videos.index:
            content = pd.DataFrame(videos.loc[vid,'content'])
            if content.empty:
                continue
            if select:
                for mood in moods:
                    if  content.loc[content['mood'] == mood].size < 1: 
                        continue
                    pick_data = content.loc[content['mood'] == mood].sample(n = 1, random_state = None).reset_index(drop=True)
                    datalist.append(pick_data)
            else:
                datalist.append(content)
    return pd.concat(datalist, ignore_index=True)

# %%
def prepare_dataset(path, num_video_per_channel = None, remove_emoji = True, select = True, seed = None):
    '''
    Returns a dataset of json files
    '''
    file_info = read_json_files(path, remove_emoji)
    data_df = data_selection(file_info, num_video_per_channel, select=select, seed=seed)
    return Dataset.from_pandas(data_df)

# %%
def get_prompt(title:str, description:str, star_num:str, mood:str) -> str:
    '''Format the instruction as a prompt for LLM.'''

    comment_type = '正面評論' if star_num.split()[1] in ['4', '5'] else '負面評論' if star_num.split()[1] in ['1', '2'] else '中立評論'
    moods = ['like','happiness','sadness','anger','fear','surprise','disgust']
    ch_moods = ['喜歡','開心','難過','生氣','害怕','驚訝','厭惡']
    if mood in moods:
        mood = ch_moods[moods.index(mood)]
    
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、詳細的回答。\
USER: 請幫這部影片生出對應需求的{comment_type}。影片標題:[{title}]。影片敘述:[{description}]。需求情感:[{mood}]。\
ASSISTANT:"


# %%
if __name__ == '__main__':
    PATH = "./train_data"
    dataset = prepare_dataset(PATH,5,seed=42)


# %%
if __name__ == '__main__':
    print(dataset[90])

# %%




# %%

# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.
import ipdb
import argparse
import json
import csv
from itertools import chain
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union, Tuple
import collections
import logging
import numpy as np
from tqdm.auto import tqdm
import os
import math
import random



import datasets
import evaluate
import torch
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy

# %%
def parse_args(str_args = None):
    parser = argparse.ArgumentParser(description="Predict answer with trained models")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the predict data."
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=32,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    
    args = parser.parse_args(str_args)
    return args

# %%
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # label_name = "label" if "label" in features[0].keys() else "labels"
        # labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        # batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch



# %%
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    output_path: str,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        #### TODO: Change to context file
        context = CONTEXT_FILE[example["relevant"]]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]
        
    return all_predictions


# %%
# Global Variable
ROOT_PATH = './'
PS_PATH = "./output_PS/"
QA_PATH = "./output_QA/"
MODEL_NAME = "pytorch_model.bin"
CONFIG_NAME = 'config.json'

with open(ROOT_PATH + 'context.json', encoding='utf-8') as f:
    CONTEXT_FILE = json.load(f)

def main(str_args):
    args = parse_args(str_args)
    # # initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps=2)
    device = accelerator.device

    # load data
    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    # load trained models, 'ps' for Paragraph Selection, 'qa' for Question Answer
    config = {}
    tokenizer = {}
    model ={}
    # load config
    config['ps'] = AutoConfig.from_pretrained(
        PS_PATH+CONFIG_NAME, trust_remote_code = False
    )
    config['qa'] = AutoConfig.from_pretrained(
        QA_PATH+CONFIG_NAME, trust_remote_code = False
    )
    # load tokenizer
    tokenizer['ps'] = AutoTokenizer.from_pretrained(
        PS_PATH, use_fast = True, trust_remote_code = False
    )
    tokenizer['qa'] = AutoTokenizer.from_pretrained(
        QA_PATH, use_fast = True, trust_remote_code = False
    )
    # load model
    model['ps'] = AutoModelForMultipleChoice.from_pretrained(
        PS_PATH+MODEL_NAME,
        config=config['ps'],
        trust_remote_code=False,
    )
    model['qa'] = AutoModelForQuestionAnswering.from_pretrained(
        QA_PATH+MODEL_NAME,
        config=config['qa'],
        trust_remote_code=False,
    )
    # Use the device given by the `accelerator` object.
    model['ps'].to(device)
    model['qa'].to(device)
        
    # column names
    paragraphs_name = "paragraphs"
    question_name = "question"
    context_name = "relevant"






    # Setup for PS
    embedding_size = model['ps'].get_input_embeddings().weight.shape[0]
    if len(tokenizer['ps']) > embedding_size:
        model['ps'].resize_token_embeddings(len(tokenizer['ps']))

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[question_name]]
        second_sentences = [
            [f"{CONTEXT_FILE[cid]}" for cid in examples[paragraphs_name][j]] for j in range(len(examples[question_name]))
        ]
        # labels = [examples[paragraphs_name][j].index(examples[label_column_name][j]) for j in range(len(examples[question_name]))]
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer['ps'](
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=False,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # processed dataset for Paragraph Selection Model
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
        )
    test_dataset = processed_datasets["test"]

    data_collator = {}
    data_collator['ps'] = DataCollatorForMultipleChoice(tokenizer['ps'])
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator['ps'], batch_size=args.per_device_batch_size)

    # Prepare Accelerator
    model['ps'], test_dataloader = accelerator.prepare(
        model['ps'], test_dataloader
    )
    # Inference for Paragraph Selection
    predict_labels = []
    model['ps'].eval()
    for step, batch in enumerate(test_dataloader):
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['token_type_ids'] = batch['token_type_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model['ps'](**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions = accelerator.gather_for_metrics(predictions)
        predict_labels += predictions.tolist()
   

    # Create the test data for QA
    # ipdb.set_trace()
    relevant_paragraph = [raw_datasets['test'][paragraphs_name][qid][label] for qid, label in enumerate(predict_labels)]
    if context_name in raw_datasets['test'].column_names:
        print("This is validation dataset. Drop original 'relevant' column.")
        raw_datasets['test'] = raw_datasets['test'].remove_columns(context_name)
    raw_datasets['test'] = raw_datasets['test'].add_column(context_name,relevant_paragraph)






    # setup for QA 
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer['qa'].padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer['qa'].model_max_length)
    # Preprocessing fucntion
    def prepare_test_features(examples):
        # ipdb.set_trace()
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_name] = [q.lstrip() for q in examples[question_name]]
        examples[context_name] = [CONTEXT_FILE[id] for id in examples[context_name]]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer['qa'](
            examples[question_name if pad_on_right else context_name],
            examples[context_name if pad_on_right else question_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding= False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    # PostProcessing function
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            output_path=args.output_path,
            n_best_size=args.n_best_size,
        )
        formatted_predictions = [{"id": k, "answer": v} for k, v in predictions.items()]
        return formatted_predictions

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat
    
    # Preprocessing
    with accelerator.main_process_first():
        qa_dataset = raw_datasets['test'].map(
            prepare_test_features,
            batched=True,
            num_proc=1,
            remove_columns=raw_datasets["test"].column_names,
            desc="Running tokenizer on test dataset",
        )
    # Load Data
    data_collator['qa'] = DataCollatorWithPadding(tokenizer['qa'])
    qa_dataset_for_model = qa_dataset.remove_columns(["example_id", "offset_mapping"])
    qa_dataloader = DataLoader(
        qa_dataset_for_model, collate_fn=data_collator['qa'], batch_size=args.per_device_batch_size
    )
    
    # Prepare Accelerator
    model['qa'], qa_dataloader = accelerator.prepare(
        model['qa'], qa_dataloader
    )

    # Predict Answer from QA
    model['qa'].eval()

    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(qa_dataloader):
        with torch.no_grad():
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            outputs = model['qa'](**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
           # if not args.pad_to_max_length: 
            if True:
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, qa_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, qa_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    predictions = post_processing_function(raw_datasets['test'], qa_dataset, outputs_numpy)
    # Output
    ### TODO: output csv
    # ipdb.set_trace()
    with open('./prediction.csv', "w",  errors='ignore', encoding='utf-8') as file:
        header = ["id","answer"]
        writer = csv.writer(file)
        writer.writerow(header)
        for answer in predictions:
            writer.writerow([answer['id'],answer['answer']])
    

# %%
if __name__ == "__main__":
    str_args = None
    # manual args, comment out if running in terminal
    str_args = [
        "--test_file", "./test.json",
        "--per_device_batch_size", "8",
        "--max_seq_length","512",
        "--n_best_size","20",
        "--doc_stride", "32",
        "--output_path", "./prediction.csv"
    ]
    main(str_args)




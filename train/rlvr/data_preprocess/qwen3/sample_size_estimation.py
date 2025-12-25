import re
import os
import json
from datasets import Dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
import argparse
from collections import defaultdict, Counter
import random


PROMPT = """You are a helpful assistant for estimating sample sizes for clinical trials. Please think step-by-step to solve the question and then generate the required sample size.

Here is the clinical trial design:
```{question}```"""


def make_prefix(dp):
    input_str = """<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>
<|im_start|>user
""" + PROMPT.format(question=dp['question'])
    input_str += """\nPlease show your entire reasoning process in **a single** <think> </think> block (do not open or close the tag more than once). Your final response must be within <answer> </answer> tags. For example,
<think>
[entire reasoning process here]
</think>
<answer>
str(sample_size_number)
</answer>.

Do not output anything after the </answer> tag.<|im_end|>
<|im_start|>assistant
"""

    return input_str


def load_sample_size_dataset(base_path):
    """
    Load sample size estimation dataset and split into train, val, test
    """
    train_path = os.path.join(base_path, 'train.jsonl')
    test_path = os.path.join(base_path, 'test.jsonl')
    
    # Load train data
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    print(f"Loaded {len(train_data)} training samples")
    
    # Randomly select 100 samples for validation
    random.seed(42)
    random.shuffle(train_data)
    
    val_data = train_data[:100]
    remaining_train_data = train_data[100:]
    
    print(f"Split into {len(remaining_train_data)} train and {len(val_data)} validation samples")
    
    # Load test data
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples")
    
    return remaining_train_data, val_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='/shared/eng/jl254/server-05/code/TrialPanorama/TrialPanorama/benchmark_data/sample_size_estimation')
    parser.add_argument('--local_dir', default='/shared/eng/jl254/server-05/code/TrialPanorama/verl/data/qwen3')
    parser.add_argument('--dataset', type=str, default='sample_size_estimation')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    train_data, val_data, test_data = load_sample_size_dataset(args.base_path)
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['answer'],
                "answer_type": example['answer_type'],
                "trial_id": example['trial_id'],
                "pmid": example['pmid']
            }
            if split == 'test':
                data_source_new = data_source + '_' + split
            else:
                data_source_new = data_source + '_' + split
            
            data = {
                "data_source": data_source_new,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "sample_size_estimation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val', data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True)
    
    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    
    # Calculate lengths
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
    
    lengths_list_val = []
    for d in val_dataset:
        lengths_list_val.append(len(d['prompt'][0]['content'].split()))

    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in val dataset: {len(val_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")
    
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list):.2f}")
    print(f"Average length of val dataset: {sum(lengths_list_val) / len(lengths_list_val):.2f}")
    print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test):.2f}")
    
    print(f"Max length of train dataset: {max(lengths_list)}")
    print(f"Max length of val dataset: {max(lengths_list_val)}")
    print(f"Max length of test dataset: {max(lengths_list_test)}")
    
    # Save to disk
    local_dir = os.path.join(args.local_dir, args.dataset)
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"\nDatasets saved to {local_dir}")
    print(f"  - train.parquet: {len(train_dataset)} samples")
    print(f"  - val.parquet: {len(val_dataset)} samples")
    print(f"  - test.parquet: {len(test_dataset)} samples")



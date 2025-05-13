"""
Utilities for loading and processing benchmark datasets.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import random

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, each representing one JSON object
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary representing the JSON object
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def load_trial_completion_data(data_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load trial completion assessment data.
    
    Args:
        data_path: Path to the trial completion assessment data directory
        
    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = os.path.join(data_path, "train.jsonl")
    test_path = os.path.join(data_path, "test.jsonl")
    
    train_data = load_jsonl(train_path) if os.path.exists(train_path) else []
    test_data = load_jsonl(test_path) if os.path.exists(test_path) else []
    
    return train_data, test_data

def load_qa_task_data(task_name: str, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load QA task data.
    
    Args:
        task_name: Name of the QA task
        data_path: Path to the QA task data JSON file, if None use default path
        
    Returns:
        List of QA samples
    """
    train_path = os.path.join(data_path, "train.jsonl")
    test_path = os.path.join(data_path, "test.jsonl")
    
    train_data = load_jsonl(train_path) if os.path.exists(train_path) else []
    test_data = load_jsonl(test_path) if os.path.exists(test_path) else []
    
    return train_data, test_data

def load_study_search_data(data_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load study search data.
    
    Args:
        data_path: Path to the study search data directory
        
    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = os.path.join(data_path, "train.jsonl")
    test_path = os.path.join(data_path, "test.jsonl")
    
    train_data = load_jsonl(train_path) if os.path.exists(train_path) else []
    test_data = load_jsonl(test_path) if os.path.exists(test_path) else []
    
    return train_data, test_data

def load_study_screening_data(data_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load study screening data.
    
    Args:
        data_path: Path to the study screening data directory
        
    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = os.path.join(data_path, "train.jsonl")
    test_path = os.path.join(data_path, "test.jsonl")
    
    train_data = load_jsonl(train_path) if os.path.exists(train_path) else []
    test_data = load_jsonl(test_path) if os.path.exists(test_path) else []
    
    return train_data, test_data

def sample_data(data: List[Dict[str, Any]], num_samples: Optional[int]) -> List[Dict[str, Any]]:
    """
    Sample a subset of the data.
    
    Args:
        data: List of data items
        num_samples: Number of samples to return, if None return all
        
    Returns:
        Sampled subset of the data
    """
    if num_samples is None or num_samples >= len(data):
        return data
    
    # Set seed for reproducibility
    random.seed(42)
    return random.sample(data, num_samples) 
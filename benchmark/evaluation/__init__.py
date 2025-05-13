"""
Evaluation metrics and utilities for benchmarking.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_classification_metrics(y_true: List[int], y_pred: List[int], y_score: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate standard classification metrics.
    
    Args:
        y_true: List of true labels (0 or 1)
        y_pred: List of predicted labels (0 or 1)
        y_score: List of prediction probabilities for the positive class (used for ROC AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add ROC AUC if scores are provided
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except:
            metrics["roc_auc"] = 0.5  # Default to random performance if calculation fails
    
    return metrics

def calculate_retrieval_metrics(
    relevant_ids: List[str], 
    retrieved_ids: List[str], 
    k_values: List[int] = [100, 200, 300, 400, 500]
) -> Dict[str, float]:
    """
    Calculate standard retrieval metrics.
    
    Args:
        relevant_ids: List of IDs that are relevant for the query
        retrieved_ids: List of IDs retrieved by the model, in rank order
        k_values: List of k values for precision@k and recall@k metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to sets for easier operations
    relevant_set = set(relevant_ids)
    
    # Initialize metrics
    metrics = {}
    
    # Calculate precision and recall at different k values
    for k in k_values:
        retrieved_at_k = retrieved_ids[:k]
        
        # Precision@k
        relevant_at_k = len(set(retrieved_at_k) & relevant_set)
        precision_at_k = relevant_at_k / k if k > 0 else 0
        metrics[f"precision@{k}"] = precision_at_k
        
        # Recall@k
        recall_at_k = relevant_at_k / len(relevant_set) if len(relevant_set) > 0 else 0
        metrics[f"recall@{k}"] = recall_at_k
        
        # F1@k
        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        else:
            f1_at_k = 0
        metrics[f"f1@{k}"] = f1_at_k
    
    # Calculate Mean Average Precision (MAP)
    ap = 0.0
    correct_count = 0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            correct_count += 1
            ap += correct_count / (i + 1)
    
    if len(relevant_set) > 0:
        metrics["map"] = ap / len(relevant_set)
    else:
        metrics["map"] = 0.0
    
    # Calculate NDCG
    metrics.update(calculate_ndcg(relevant_ids, retrieved_ids, k_values))
    
    return metrics

def calculate_ndcg(
    relevant_ids: List[str], 
    retrieved_ids: List[str], 
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        relevant_ids: List of IDs that are relevant for the query
        retrieved_ids: List of IDs retrieved by the model, in rank order
        k_values: List of k values for NDCG@k metrics
        
    Returns:
        Dictionary of NDCG metrics
    """
    relevant_set = set(relevant_ids)
    
    # Initialize metrics
    metrics = {}
    
    # Calculate DCG and IDCG for different k values
    for k in k_values:
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                # Using binary relevance (1 for relevant, 0 for not relevant)
                # DCG formula: sum(rel_i / log2(i+2))
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        ideal_rank = min(k, len(relevant_set))
        for i in range(ideal_rank):
            idcg += 1.0 / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg
    
    return metrics

# Import specific evaluation modules
from .search_metrics import evaluate_search_results
from .screening_metrics import evaluate_screening_results
from .completion_metrics import evaluate_completion_results 
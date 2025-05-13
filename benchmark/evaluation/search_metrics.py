"""
Evaluation metrics for the study search task.
"""

from typing import Dict, List, Any, Optional, Union
from collections import Counter, defaultdict
import pdb

def evaluate_search_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for the study search task.
    
    Args:
        results: List of results from the benchmark
        
    Returns:
        Dictionary of metrics
    """
    # Extract evaluation data
    evaluations = [r.get("evaluation", {}) for r in results if "evaluation" in r]
    
    # Filter out samples with errors
    valid_evaluations = [e for e in evaluations if "error" not in e]
    
    if not valid_evaluations:
        return {
            "error": "No valid evaluations found",
            "num_samples": len(results),
            "num_errors": len(results) - len(valid_evaluations)
        }
    
    # Aggregate metrics across all queries
    all_precision_at_k = defaultdict(list)
    all_recall_at_k = defaultdict(list)

    for eval_result in valid_evaluations:
        metrics = eval_result.get("metrics", {})
        
        # Collect precision at different k values
        for metric_name, metric_values in metrics.items():
            if metric_name.startswith("precision@"):
                k = int(metric_name.split("@")[1])
                all_precision_at_k[k].append(metric_values)
            elif metric_name.startswith("recall@"):
                k = int(metric_name.split("@")[1])
                all_recall_at_k[k].append(metric_values)
    
    # Calculate average metrics
    avg_metrics = {
        "num_samples": len(results),
        "num_valid": len(valid_evaluations),
        "num_errors": len(results) - len(valid_evaluations),
    }
    
    # average precision, recall
    for k, v in all_precision_at_k.items():
        avg_metrics[f"avg_precision@{k}"] = sum(v) / len(v)
    for k, v in all_recall_at_k.items():
        avg_metrics[f"avg_recall@{k}"] = sum(v) / len(v)
    
    return avg_metrics 
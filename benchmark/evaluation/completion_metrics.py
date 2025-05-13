"""
Evaluation metrics for the trial completion assessment task.
"""

from typing import Dict, List, Any, Optional, Union
from collections import Counter

from . import calculate_classification_metrics

def evaluate_completion_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for the trial completion assessment task.
    
    Args:
        results: List of results from the benchmark
        
    Returns:
        Dictionary of metrics
    """
    # Extract evaluation data
    evaluations = [r.get("evaluation", {}) for r in results if "evaluation" in r]
    
    # Filter out samples with errors
    valid_evaluations = [e for e in evaluations if "error" not in e and e.get("pred_outcome") is not None]
    
    if not valid_evaluations:
        return {
            "error": "No valid evaluations found",
            "num_samples": len(results),
            "num_errors": len(results) - len(valid_evaluations)
        }
    
    # Binary classification metrics for outcome prediction
    y_true_outcome = [1 if e.get("true_outcome") == "complete" else 0 for e in valid_evaluations]
    y_pred_outcome = [1 if e.get("pred_outcome") == "complete" else 0 for e in valid_evaluations]
    
    outcome_metrics = calculate_classification_metrics(y_true_outcome, y_pred_outcome)
    
    # Calculate termination type prediction metrics
    terminated_samples = [e for e in valid_evaluations 
                          if e.get("true_outcome") == "terminate" and e.get("pred_outcome") == "terminate"]
    
    if terminated_samples:
        terminate_type_accuracy = sum(1 for e in terminated_samples if e.get("correct_terminate_type")) / len(terminated_samples)
        
        # Count by termination type
        true_terminate_types = Counter([e.get("true_terminate_type") for e in terminated_samples])
        pred_terminate_types = Counter([e.get("pred_terminate_type") for e in terminated_samples])
        
        # Calculate accuracy per termination type
        type_accuracies = {}
        for term_type, count in true_terminate_types.items():
            if term_type:
                correct = sum(1 for e in terminated_samples 
                             if e.get("true_terminate_type") == term_type and e.get("pred_terminate_type") == term_type)
                type_accuracies[term_type] = correct / count if count > 0 else 0
    else:
        terminate_type_accuracy = None
        true_terminate_types = {}
        pred_terminate_types = {}
        type_accuracies = {}
    
    # Combine all metrics
    metrics = {
        "num_samples": len(results),
        "num_valid": len(valid_evaluations),
        "num_errors": len(results) - len(valid_evaluations),
        "outcome_prediction": outcome_metrics,
        "termination_type": {
            "accuracy": terminate_type_accuracy,
            "true_type_distribution": true_terminate_types,
            "predicted_type_distribution": pred_terminate_types,
            "accuracy_by_type": type_accuracies
        }
    }
    
    return metrics 
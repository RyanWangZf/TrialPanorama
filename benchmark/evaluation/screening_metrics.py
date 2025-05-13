"""
Evaluation metrics for study screening tasks.
"""
import pdb
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_screening_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate the results of a study screening task.
    
    Args:
        results: List of benchmark result objects, each containing sample, prediction, and evaluation
        
    Returns:
        Dictionary of aggregate metrics
    """
    # Extract true and predicted labels from results
    y_true = []
    y_pred = []
    confidence_scores = []
    
    for result in results:
        evaluation = result.get('evaluation', {})
        true_label = evaluation.get('true_label', '')
        pred_label = evaluation.get('pred_label', '')
        confidence = evaluation.get('confidence', 0.5)
        
        # Skip samples where we couldn't get a valid prediction
        if not true_label or not pred_label:
            continue
        
        # Convert labels to binary (1 for include, 0 for exclude)
        y_true.append(1 if true_label == 'included' else 0)
        y_pred.append(1 if pred_label == 'included' else 0)
        confidence_scores.append(confidence)
    
    # Calculate metrics
    metrics = {
        'num_samples': len(y_true),
        'num_included': sum(y_true),
        'num_excluded': len(y_true) - sum(y_true),
        'accuracy': accuracy_score(y_true, y_pred) if y_true else 0.0,
        'precision': precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        'recall': recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
        'f1': f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0,
    }
    
    # Calculate average confidence
    if confidence_scores:
        metrics['avg_confidence'] = np.mean(confidence_scores)
        
        # Calculate average confidence for correct and incorrect predictions
        correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
        incorrect_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
        
        if correct_indices:
            metrics['avg_confidence_correct'] = np.mean([confidence_scores[i] for i in correct_indices])
        else:
            metrics['avg_confidence_correct'] = 0.0
            
        if incorrect_indices:
            metrics['avg_confidence_incorrect'] = np.mean([confidence_scores[i] for i in incorrect_indices])
        else:
            metrics['avg_confidence_incorrect'] = 0.0
    
    # Calculate confusion matrix elements
    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            'true_positives': int(tp),  # Correctly included
            'true_negatives': int(tn),  # Correctly excluded
            'false_positives': int(fp), # Incorrectly included
            'false_negatives': int(fn)  # Incorrectly excluded
        })
    
    # Calculate specificity (true negative rate)
    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['specificity'] = specificity
    
    return metrics 
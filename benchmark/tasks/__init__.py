"""
Task implementations for clinical trial benchmark.
"""

from typing import Dict, List, Any, Optional, Union

class Task:
    """Base class for all tasks."""
    
    def __init__(self, name: str):
        """
        Initialize the task.
        
        Args:
            name: Name of the task
        """
        self.name = name
    
    def prepare_input(self, sample: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Prepare the input for the model based on the task.
        
        Args:
            sample: Sample from the dataset
            
        Returns:
            Formatted input for the model
        """
        raise NotImplementedError("Subclasses must implement prepare_input()")
    
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
        """
        Evaluate the model's prediction for a sample.
        
        Args:
            sample: Sample from the dataset
            prediction: Model's prediction
            
        Returns:
            Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate_prediction()")
    
    def run_benchmark(self, model, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Run the benchmark for the given model and data.
        
        Args:
            model: Model to evaluate
            data: Data to evaluate on
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark results
        """
        raise NotImplementedError("Subclasses must implement run_benchmark()")

from .study_search import StudySearchTask
from .study_screening import StudyScreeningTask
from .trial_completion import TrialCompletionAssessmentTask
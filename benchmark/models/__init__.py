"""
Model interfaces for benchmarking.
"""

from typing import Dict, List, Any, Optional, Union

class ModelInterface:
    """Base interface for all models."""
    
    def __init__(self, model_name: str):
        """
        Initialize the model interface.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
    
    def predict(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Make a prediction for the given input.
        
        Args:
            input_data: Input data for the model
            **kwargs: Additional arguments for the prediction
            
        Returns:
            Model prediction
        """
        raise NotImplementedError("Subclasses must implement predict()")

from .llm_azure import AzureOpenAIModel
from .pubmed_api import PubMedAPI 
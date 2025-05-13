"""
Study search task implementation.
"""
import os
import time
import json
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .. import config, data
from ..evaluation import calculate_retrieval_metrics
from ..evaluation.search_metrics import evaluate_search_results
from . import Task
from ..models.pubmed_api import PubMedAPI

class StudySearchTask(Task):
    """Task for searching relevant clinical trials given a query."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the study search task.
        
        Args:
            system_prompt: System prompt to use for the model
        """
        super().__init__("study_search")
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.pubmed_api = None
        
        # Initialize PubMed API for executing search queries
        try:
            # Only PUBMED_API_KEY is needed, which is optional
            api_key = os.environ.get("PUBMED_API_KEY")
            
            # Initialize with just the API key - email and tool are now optional
            self.pubmed_api = PubMedAPI(api_key=api_key)
            
            if api_key:
                print("PubMed API initialized with API key (10 requests per second)")
            else:
                print("PubMed API initialized without API key (3 requests per second)")
                
        except Exception as e:
            print(f"Warning: Could not initialize PubMed API: {e}")
            print("String predictions will not be evaluated with PubMed search")
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the study search task."""
        return """Given the systematic review setup, formulate an effective search query that will retrieve relevant clinical trials.

Your output should be a JSON object with the following fields:
- query: The search query that will be used by PubMed API to fetch the results
"""
    
    def prepare_input(self, sample: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Prepare the search query as input for the model.
        
        Args:
            sample: Search query data
            
        Returns:
            Formatted input for the model
        """
        # Extract the query information
        systematic_review_setup = [f"# {key}: {value}" for key, value in sample.get("inputs", {}).items()]
        systematic_review_setup = "\n".join(systematic_review_setup)

        # For now, assume the sample already contains a structured query
        # A real implementation would need to format the query appropriately
        return systematic_review_setup
    
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
        """
        Evaluate the model's search results against the gold standard.
        
        Args:
            sample: Sample containing the query and relevant documents
            prediction: Model's search results (list of documents or IDs)
            
        Returns:
            Evaluation metrics
        """
        # Extract relevant IDs from the sample
        relevant_ids = sample.get("labels", [])

        # parse the prediction
        query = prediction.get("query", "")
        
        # If prediction is a string (e.g., a search query), execute it with PubMed API
        if self.pubmed_api:
            try:
                # Execute the search query using PubMed API
                search_results = self.pubmed_api.search(query, max_results=1000, return_ids_only=True)
                retrieved_ids = [doc.get("id", "") for doc in search_results]
                print(f"Executed PubMed search with query: '{query}'")
                print(f"Retrieved {len(retrieved_ids)} results")
            except Exception as e:
                print(f"Error executing PubMed search: {e}")
                retrieved_ids = []
        else:
            print("Warning: Received query string but no PubMed API client available")
            retrieved_ids = []

        # Calculate retrieval metrics
        metrics = calculate_retrieval_metrics(relevant_ids, retrieved_ids, k_values=[100, 200, 300, 400, 500])
        
        return {
            "query": sample.get("query", ""),
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics
        }
    
    def run_benchmark(
        self, 
        model, 
        data: List[Dict[str, Any]], 
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the study search benchmark for the given model and data.
        
        Args:
            model: Model to evaluate
            data: Search queries to evaluate on
            output_path: Path to save the results
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark results
        """
        results = []
        
        # If model is PubMedAPI, use it for executing searches directly
        if isinstance(model, PubMedAPI) and self.pubmed_api is None:
            self.pubmed_api = model
        
        print(f"Running study search benchmark on {len(data)} queries...")
        
        for sample in tqdm(data):
            # Prepare input for the model
            input_data = self.prepare_input(sample)
            
            # Make prediction
            try:
                prediction = model.predict(
                    input_data, 
                    system_prompt=self.system_prompt,
                    output_format="json"
                )
                
                # Evaluate prediction
                evaluation = self.evaluate_prediction(sample, prediction)
                results.append({
                    "sample": sample,
                    "input": input_data,
                    "prediction": prediction,
                    "evaluation": evaluation
                })
                
            except Exception as e:
                print(f"Error for query {sample.get('query_id', '')}: {e}")
                # Add failed sample to results
                results.append({
                    "sample": sample,
                    "input": input_data,
                    "prediction": str(e),
                    "evaluation": {
                        "query": sample.get("query", ""),
                        "error": str(e)
                    }
                })
                    
        # Calculate aggregate metrics
        metrics = evaluate_search_results(results)
        
        # Save results if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save detailed results
            with open(output_path, "w") as f:
                json.dump({
                    "task": self.name,
                    "model": model.model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics,
                    "results": results
                }, f, indent=2)
            
            # Save summary metrics to a separate file
            summary_path = os.path.join(os.path.dirname(output_path), f"{model.model_name}_summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "task": self.name,
                    "model": model.model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics
                }, f, indent=2)
            
            print(f"Results saved to {output_path}")
            print(f"Summary metrics saved to {summary_path}")
        
        return {
            "metrics": metrics,
            "results": results
        } 
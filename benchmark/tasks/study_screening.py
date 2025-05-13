"""
Study screening task implementation.
"""
import pdb
import os
import time
import json
import copy
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .. import data
from ..evaluation import calculate_classification_metrics
from ..evaluation.screening_metrics import evaluate_screening_results
from . import Task

class StudyScreeningTask(Task):
    """Task for screening clinical trials for inclusion in a systematic review."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the study screening task.
        
        Args:
            system_prompt: System prompt to use for the model
        """
        super().__init__("study_screening")
        
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the study screening task."""
        return """You are an expert in systematic reviews of clinical trials. Your task is to determine whether each candidate study should be included or excluded from a systematic review based on the systematic review setup provided.

For each candidate study in the list, analyze the title and abstract carefully, considering:
- Whether the study meets the research question and goals of the systematic review
- Whether the study matches the PICO criteria (Population, Intervention, Comparison, Outcome)
- Whether the study design is appropriate for inclusion
- Any other inclusion or exclusion criteria specified in the systematic review setup

For each study, provide your decision as "included" or "excluded", along with a confidence score (0-100).

Your response must be a JSON object with a "results" array containing one object for each candidate study, in the same order as provided:
{
  "results": [
    {
      "PMID": "study_pmid_1",
      "decision": "included" or "excluded",
      "confidence": [0-100 integer]
    },
    {
      "PMID": "study_pmid_2",
      "decision": "included" or "excluded",
      "confidence": [0-100 integer]
    },
    ...
  ]
}"""
    
    def prepare_input(self, sample: Dict[str, Any]) -> str:
        """
        Prepare the systematic review setup and candidate studies as input for the model.
        
        Args:
            sample: Screening sample data
            
        Returns:
            Formatted input for the model
        """
        # Format systematic review setup
        systematic_review_setup = [f"# {key}: {value}" for key, value in sample.get("inputs", {}).items()]
        systematic_review_setup = "\n".join(systematic_review_setup)
        
        # Format candidate studies
        candidates = sample.get("context", [])
        studies_text = []
        
        for i, study in enumerate(candidates):
            pmid = study.get("PMID", "")
            title = study.get("title", "")
            abstract = study.get("abstract", "")
            
            study_text = f"## Study {i+1} (PMID: {pmid})\n"
            study_text += f"### Title\n{title}\n\n"
            study_text += f"### Abstract\n{abstract}\n"
            
            studies_text.append(study_text)
        
        # Combine all parts
        input_text = "# Systematic Review Setup\n"
        input_text += systematic_review_setup
        input_text += "\n\n# Candidate Studies\n"
        input_text += "\n\n".join(studies_text)
        
        return input_text
    
    def parse_prediction(self, prediction: Any, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse the model prediction into a standard format.
        
        Args:
            prediction: Model prediction
            sample: Original sample for reference
            
        Returns:
            List of standardized prediction objects
        """
        # Get the list of PMIDs from context for reference
        context = sample.get("context", [])
        pmids = [study.get("PMID", "") for study in context]
        
        # Default result if parsing fails - mark all as "excluded"
        default_results = [
            {"PMID": pmid, "decision": "excluded", "confidence": 0.5} 
            for pmid in pmids
        ]
        
        # Try to parse the prediction
        if isinstance(prediction, dict):
            # Check if the prediction has a "results" key containing the list of decisions
            if "results" in prediction and isinstance(prediction["results"], list):
                pred_list = prediction["results"]
                if all(isinstance(p, dict) and "PMID" in p and "decision" in p for p in pred_list):
                    parsed = copy.deepcopy(pred_list)
                    
                    # Normalize decisions and convert confidence to 0-1 scale
                    for p in parsed:
                        p["decision"] = p.get("decision", "").lower()
                        p["confidence"] = float(p.get("confidence", 50)) / 100.0
                    
                    return parsed
        elif isinstance(prediction, list):
            # Already a list, verify it has the expected structure
            if all(isinstance(p, dict) and "PMID" in p and "decision" in p for p in prediction):
                parsed = copy.deepcopy(prediction)
                
                # Normalize decisions and convert confidence to 0-1 scale
                for p in parsed:
                    p["decision"] = p.get("decision", "").lower()
                    p["confidence"] = float(p.get("confidence", 50)) / 100.0
                
                return parsed
        elif isinstance(prediction, str):
            # Try to parse as JSON
            try:
                pred_obj = json.loads(prediction)
                # Check if it's a dict with a "results" key
                if isinstance(pred_obj, dict) and "results" in pred_obj and isinstance(pred_obj["results"], list):
                    pred_list = pred_obj["results"]
                elif isinstance(pred_obj, list):
                    pred_list = pred_obj
                else:
                    return default_results
                
                parsed = []
                
                # Process each prediction in the list
                for p in pred_list:
                    if isinstance(p, dict) and "PMID" in p and "decision" in p:
                        parsed_item = {
                            "PMID": p.get("PMID", ""),
                            "decision": p.get("decision", "").lower(),
                            "confidence": float(p.get("confidence", 50)) / 100.0
                        }
                        parsed.append(parsed_item)
                
                if parsed:
                    return parsed
            except Exception as e:
                print(f"Error parsing prediction: {e}")
        
        # If we get here, parsing failed
        print("Warning: Could not parse model prediction. Using default result.")
        return default_results
    
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
        """
        Evaluate the model's predictions for a batch of screening decisions.
        
        Args:
            sample: Sample from the dataset
            prediction: Model's predictions
            
        Returns:
            Evaluation metrics and details
        """

        # Get the true labels
        true_labels = {item["PMID"]: item["label"] for item in sample.get("labels", [])}
        
        # Parse the predictions
        parsed_predictions = self.parse_prediction(prediction, sample)
        
        # Evaluate each prediction
        evaluations = []
        is_correct_list = []
        
        for pred in parsed_predictions:
            pmid = pred.get("PMID", "")
            pred_label = pred.get("decision", "")
            confidence = pred.get("confidence", 0.5)
            
            # Get the true label for this PMID
            true_label = true_labels.get(pmid, "")
            
            # Determine if prediction is correct
            is_correct = pred_label == true_label if pred_label and true_label else None
            if is_correct is not None:
                is_correct_list.append(is_correct)
            
            evaluations.append({
                "PMID": pmid,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "is_correct": is_correct
            })
        
        # Calculate micro accuracy (per study)
        micro_accuracy = sum(is_correct_list) / len(is_correct_list) if is_correct_list else 0.0
        
        # Calculate macro accuracy (per review - all studies must be correct)
        macro_accuracy = 1.0 if all(is_correct_list) else 0.0
        
        return {
            "review_id": sample.get("metadata", {}).get("PMID", ""),
            "evaluations": evaluations,
            "micro_accuracy": micro_accuracy,
            "macro_accuracy": macro_accuracy
        }
    
    def run_benchmark(
        self, 
        model, 
        data: List[Dict[str, Any]], 
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the study screening benchmark for the given model and data.
        
        Args:
            model: Model to evaluate
            data: Screening samples to evaluate on
            output_path: Path to save the results
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark results
        """
        results = []
        
        print(f"Running study screening benchmark on {len(data)} systematic reviews...")
        
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
                
                # Log the prediction format for debugging
                if isinstance(prediction, dict):
                    if "results" in prediction:
                        print(f"✓ Prediction format correct: contains 'results' key with {len(prediction['results'])} items")
                    else:
                        print(f"⚠ Prediction format unexpected: dictionary without 'results' key")
                elif isinstance(prediction, list):
                    print(f"⚠ Prediction format unexpected: direct list with {len(prediction)} items")
                elif isinstance(prediction, str):
                    print(f"⚠ Prediction format as string (length: {len(prediction)})")
                else:
                    print(f"⚠ Prediction format unknown: {type(prediction)}")
                
                # Evaluate prediction
                evaluation = self.evaluate_prediction(sample, prediction)
                results.append({
                    "sample_id": sample.get("metadata", {}).get("PMID", ""),
                    "input": input_data,
                    "prediction": prediction,
                    "evaluation": evaluation
                })
                
            except Exception as e:
                print(f"Error for review {sample.get('metadata', {}).get('PMID', '')}: {e}")
                # Add failed sample to results
                results.append({
                    "sample_id": sample.get("metadata", {}).get("PMID", ""),
                    "input": input_data,
                    "prediction": str(e),
                    "evaluation": {
                        "review_id": sample.get("metadata", {}).get("PMID", ""),
                        "evaluations": [],
                        "micro_accuracy": 0.0,
                        "macro_accuracy": 0.0,
                        "error": str(e)
                    }
                })
        
        # Calculate aggregate metrics
        micro_accuracies = [r["evaluation"].get("micro_accuracy", 0) for r in results]
        macro_accuracies = [r["evaluation"].get("macro_accuracy", 0) for r in results]
        
        metrics = {
            "num_reviews": len(results),
            "avg_micro_accuracy": sum(micro_accuracies) / len(micro_accuracies) if micro_accuracies else 0.0,
            "avg_macro_accuracy": sum(macro_accuracies) / len(macro_accuracies) if macro_accuracies else 0.0
        }
        
        # Calculate traditional metrics across all studies
        all_evaluations = []
        for r in results:
            all_evaluations.extend(r["evaluation"].get("evaluations", []))
        
        if all_evaluations:
            screening_metrics = evaluate_screening_results(
                [{"evaluation": eval_item} for eval_item in all_evaluations]
            )
            metrics.update(screening_metrics)
        
        # Print summary metrics
        print("\nScreening Results:")
        print(f"Number of Reviews: {metrics['num_reviews']}")
        print(f"Average Micro Accuracy (per study): {metrics['avg_micro_accuracy']:.4f}")
        print(f"Average Macro Accuracy (per review): {metrics['avg_macro_accuracy']:.4f}")
        
        if "accuracy" in metrics:
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        
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
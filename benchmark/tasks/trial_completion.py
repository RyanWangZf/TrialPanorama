"""
Trial completion assessment task implementation.
"""

import os
import time
import json
import copy
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .. import data
from ..evaluation import calculate_classification_metrics
from ..evaluation.completion_metrics import evaluate_completion_results
from . import Task

class TrialCompletionAssessmentTask(Task):
    """Task for assessing whether a clinical trial will complete or terminate."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the trial completion assessment task.
        
        Args:
            system_prompt: System prompt to use for the model
        """
        super().__init__("trial_completion")
        
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the trial completion assessment task."""
        return """You are an expert in clinical trials. Your task is to predict whether a given clinical trial will successfully complete or terminate early, based on the information provided.

Analyze the trial information carefully, considering:
- Study population and eligibility criteria
- Intervention details and complexity
- Study design and duration
- Outcome measures
- Sample size and recruitment targets
- Sponsorship and funding
- Any other factors that might influence trial completion

For each trial, provide your prediction as "complete" or "terminate" based on your analysis.

If you predict the trial will terminate, also provide the likely reason for termination from one of these categories:
- "enrollment issues"
- "feasibility"
- "lack of efficacy"
- "regulatory/approval"
- "safety issues"

Your response should be in JSON format:
{
  "outcome": "complete" or "terminate",
  "confidence": [0-100 integer],
  "termination_reason": "[one of the reasons above, only if outcome is terminate]",
  "explanation": "Brief explanation of your prediction"
}"""
    
    def prepare_input(self, sample: Dict[str, Any]) -> str:
        """
        Prepare the trial information as input for the model.
        
        Args:
            sample: Trial data
            
        Returns:
            Formatted input for the model
        """
        # Extract metadata, inputs, and context
        metadata = sample.get("metadata", {})
        inputs = sample.get("inputs", {})
        context = sample.get("context", [])
        
        # Format the input as structured text
        sections = []
        
        # Add trial identifier and title
        nct_id = metadata.get("study_nctid", "Unknown")
        title = inputs.get("title", "Unknown Title")
        sections.append(f"# Trial: {nct_id}\n## Title\n{title}")
        
        # Add study design information from inputs
        sections.append("# Study Design")
        design_info = []
        for key, value in inputs.items():
            if key != "title" and value is not None:  # Skip title and None values
                if key == "number_of_arms":
                    # Format as integer if it's a float
                    value = int(value) if isinstance(value, float) and value.is_integer() else value
                design_info.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        sections.append("\n".join(design_info))
        
        # Process context data
        for item in context:
            item_type = item.get("type", "")
            
            if item_type == "eligibility_criteria":
                sections.append("# Eligibility Criteria")
                criteria = item.get("criteria", "").replace("~", "\n")  # Replace ~ with newlines
                gender = item.get("gender", "")
                min_age = item.get("minimum_age", "")
                max_age = item.get("maximum_age", "")
                
                age_gender = []
                if gender:
                    age_gender.append(f"Gender: {gender}")
                if min_age or max_age:
                    age_gender.append(f"Age: {min_age} to {max_age}")
                
                if age_gender:
                    sections.append("\n".join(age_gender))
                sections.append(criteria)
            
            elif item_type == "arm_design":
                sections.append("# Study Arms")
                for arm in item.get("arms", []):
                    arm_title = arm.get("title", "")
                    arm_type = arm.get("group_type", "")
                    arm_desc = arm.get("description", "")
                    sections.append(f"## {arm_title} ({arm_type})\n{arm_desc}")
        
        # Combine all sections
        input_text = "\n\n".join(sections)
        
        return input_text
    
    def parse_prediction(self, prediction: Any) -> Dict[str, Any]:
        """
        Parse the model prediction into a standard format.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Standardized prediction
        """
        # Default values
        parsed = {
            "outcome": None,
            "confidence": 0.5,
            "termination_reason": None,
            "explanation": ""
        }
        
        # Try to parse the prediction
        if isinstance(prediction, dict):
            # If prediction is already a dict, extract the fields
            parsed["outcome"] = prediction.get("outcome", "").lower()
            parsed["confidence"] = float(prediction.get("confidence", 50)) / 100.0  # Convert 0-100 to 0-1
            parsed["termination_reason"] = prediction.get("termination_reason", "").lower() if parsed["outcome"] == "terminate" else None
            parsed["explanation"] = prediction.get("explanation", "")
        
        elif isinstance(prediction, str):
            # Try to parse as JSON
            try:
                pred_dict = json.loads(prediction)
                parsed["outcome"] = pred_dict.get("outcome", "").lower()
                parsed["confidence"] = float(pred_dict.get("confidence", 50)) / 100.0
                parsed["termination_reason"] = pred_dict.get("termination_reason", "").lower() if parsed["outcome"] == "terminate" else None
                parsed["explanation"] = pred_dict.get("explanation", "")
            except Exception as e:
                # If parsing fails, check for "complete" or "terminate" in the text
                prediction_lower = prediction.lower()
                if "complete" in prediction_lower:
                    parsed["outcome"] = "complete"
                elif "terminate" in prediction_lower:
                    parsed["outcome"] = "terminate"
                    
                    # Try to extract termination reason
                    for reason in ["insufficient_enrollment", "safety_concerns", "lack_of_efficacy", 
                                 "funding_issues", "design_flaws", "external_factors"]:
                        if reason in prediction_lower:
                            parsed["termination_reason"] = reason
                            break
                
                # Take the whole text as explanation
                parsed["explanation"] = prediction
        
        return parsed
    
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
        """
        Evaluate the model's prediction for trial completion assessment.
        
        Args:
            sample: Sample from the dataset
            prediction: Model's prediction
            
        Returns:
            Evaluation metrics
        """
        # Get the true outcome and termination reason from labels
        labels = sample.get("labels", {})
        true_outcome = labels.get("outcome", "").lower()
        true_terminate_type = labels.get("terminate_type", "").lower() if true_outcome == "terminate" else None
        true_terminate_desc = labels.get("terminate_reason_desc", "")
        
        # Parse the prediction
        parsed_prediction = self.parse_prediction(prediction)
        pred_outcome = parsed_prediction["outcome"]
        pred_terminate_type = parsed_prediction["termination_reason"]
        confidence = parsed_prediction["confidence"]
        
        # Determine if the outcome prediction is correct
        is_outcome_correct = pred_outcome == true_outcome if pred_outcome and true_outcome else None
        
        # Determine if the termination type prediction is correct (only relevant if both are "terminate")
        is_terminate_type_correct = None
        if true_outcome == "terminate" and pred_outcome == "terminate":
            is_terminate_type_correct = pred_terminate_type == true_terminate_type
        
        return {
            "nct_id": sample.get("metadata", {}).get("study_nctid", ""),
            "true_outcome": true_outcome,
            "pred_outcome": pred_outcome,
            "true_terminate_type": true_terminate_type,
            "true_terminate_desc": true_terminate_desc,
            "pred_terminate_type": pred_terminate_type,
            "confidence": confidence,
            "explanation": parsed_prediction["explanation"],
            "correct_outcome": is_outcome_correct,
            "correct_terminate_type": is_terminate_type_correct
        }
    
    def run_benchmark(
        self, 
        model, 
        data: List[Dict[str, Any]], 
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the trial completion assessment benchmark for the given model and data.
        
        Args:
            model: Model to evaluate
            data: Trial samples to evaluate on
            output_path: Path to save the results
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark results
        """
        results = []
        
        print(f"Running trial completion assessment benchmark on {len(data)} trials...")
        
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
                    "sample_id": sample.get("metadata", {}).get("study_nctid", ""),
                    "input": input_data,
                    "prediction": prediction,
                    "evaluation": evaluation
                })
                
            except Exception as e:
                print(f"Error for trial {sample.get('metadata', {}).get('study_nctid', '')}: {e}")
                # Add failed sample to results
                results.append({
                    "sample_id": sample.get("metadata", {}).get("study_nctid", ""),
                    "input": input_data,
                    "prediction": str(e),
                    "evaluation": {
                        "nct_id": sample.get("metadata", {}).get("study_nctid", ""),
                        "true_outcome": sample.get("labels", {}).get("outcome", ""),
                        "pred_outcome": None,
                        "error": str(e)
                    }
                })

            # break # debug
        
        # Calculate aggregate metrics
        metrics = evaluate_completion_results(results)
        
        # Print summary metrics
        print("\nTrial Completion Assessment Results:")
        print(f"Number of Trials: {metrics.get('num_samples', 0)}")
        print(f"Valid Evaluations: {metrics.get('num_valid', 0)}")
        
        outcome_metrics = metrics.get('outcome_prediction', {})
        if outcome_metrics:
            print(f"\nOutcome Prediction Metrics:")
            print(f"Accuracy: {outcome_metrics.get('accuracy', 0):.4f}")
            print(f"Precision: {outcome_metrics.get('precision', 0):.4f}")
            print(f"Recall: {outcome_metrics.get('recall', 0):.4f}")
            print(f"F1 Score: {outcome_metrics.get('f1', 0):.4f}")
        
        termination_metrics = metrics.get('termination_type', {})
        if termination_metrics and termination_metrics.get('accuracy') is not None:
            print(f"\nTermination Type Prediction Accuracy: {termination_metrics.get('accuracy', 0):.4f}")
        
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
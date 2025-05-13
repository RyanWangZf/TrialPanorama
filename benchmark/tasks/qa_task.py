"""
QA task implementation for clinical trial related questions.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from benchmark.data import load_qa_task_data
from benchmark.evaluation import calculate_classification_metrics
from benchmark.tasks import Task

class QATask(Task):
    """Task for answering questions about clinical trials."""
    
    def __init__(self, task_name: str, system_prompt: Optional[str] = None):
        """
        Initialize the QA task.
        
        Args:
            task_name: Specific QA task name (design_arms_qa, design_criteria_qa, design_outcome_qa, review_qa, sample_size_estimation)
            system_prompt: System prompt to use for the model
        """
        super().__init__(task_name)
        
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the QA task."""
        if self.name in ["arm_design", "eligibility_criteria_design", "endpoint_design"]:
            return "You are a helpful assistant for designing clinical trials. Please output in JSON formatted exactly as: ```json\n{\"step_by_step_thinking\": Str, \"answer_choice\": \"A\"|\"B\"|\"C\"|...}\n```. Do not include anything else in your response."
        elif self.name == "evidence_summary":
            return "You are a helpful assistant for answering biomedical questions with your knowledge of clinical trials. Please output in JSON formatted exactly as: ```json\n{\"step_by_step_thinking\": Str, \"answer_choice\": \"A\"|\"B\"|\"C\"|...}\n```. Do not include anything else in your response."
        elif self.name == "sample_size_estimation":
            return "You are a helpful assistant for estimating the sample size for the given clinical trial. Please output in JSON formatted exactly as: ```json\n{\"step_by_step_thinking\": Str, \"sample_size_needed\": Int}\n```. Do not include anything else in your response."
        else:
            return "You are a helpful assistant for answering questions about clinical trials."
    
    def prepare_input(self, sample: Dict[str, Any]) -> str:
        """
        Prepare the question and options as input for the model.
        
        Args:
            sample: Question data
            
        Returns:
            Formatted input for the model
        """
        if self.name == "sample_size_estimation":
            return sample["question"]
        else:
            user_prompt = sample["question"] + "\n"
            for k, v in sample["options"].items():
                user_prompt += f"{k}. {v}\n"
            return user_prompt
    
    def parse_prediction(self, prediction: Any) -> Dict[str, Any]:
        """
        Parse the model prediction into a standard format.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Standardized prediction
        """
        # Try to extract JSON from the response
        prediction = prediction.strip()
        if "```json" in prediction:
            try:
                json_part = prediction.split("```json")[1].split("```")[0].strip()
                return json.loads(json_part)
            except:
                pass
        
        try:
            return json.loads(prediction)
        except:
            # If parsing fails, return the raw prediction
            return {"raw_response": prediction}
    
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
        """
        Evaluate the model's prediction for QA tasks.
        
        Args:
            sample: Sample from the dataset
            prediction: Model's prediction
            
        Returns:
            Evaluation metrics
        """
        parsed_prediction = self.parse_prediction(prediction)
        
        if self.name == "sample_size_estimation":
            try:
                pred_sample_size = parsed_prediction.get("sample_size_needed")
                true_sample_size = sample.get("answer")
                
                # Check if the prediction is within 20% of the true value
                is_correct = False
                if pred_sample_size and true_sample_size:
                    pred_sample_size = int(pred_sample_size)
                    true_sample_size = int(true_sample_size)
                    is_correct = 0.8 * true_sample_size <= pred_sample_size <= 1.2 * true_sample_size
                
                return {
                    "id": sample.get("id", ""),
                    "true_answer": true_sample_size,
                    "pred_answer": pred_sample_size,
                    "correct": is_correct,
                    "thinking": parsed_prediction.get("step_by_step_thinking", "")
                }
            except:
                return {
                    "id": sample.get("id", ""),
                    "true_answer": sample.get("answer"),
                    "pred_answer": None,
                    "correct": False,
                    "error": "Failed to parse prediction"
                }
        else:
            try:
                pred_answer = parsed_prediction.get("answer_choice", "").lower()
                true_answer = sample.get("answer", "").lower()
                
                return {
                    "id": sample.get("id", ""),
                    "true_answer": true_answer,
                    "pred_answer": pred_answer,
                    "correct": pred_answer == true_answer,
                    "thinking": parsed_prediction.get("step_by_step_thinking", "")
                }
            except:
                return {
                    "id": sample.get("id", ""),
                    "true_answer": sample.get("answer"),
                    "pred_answer": None,
                    "correct": False,
                    "error": "Failed to parse prediction"
                }
    
    def run_benchmark(
        self, 
        model, 
        data: List[Dict[str, Any]], 
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the QA benchmark for the given model and data.
        
        Args:
            model: Model to evaluate
            data: Question samples to evaluate on
            output_path: Path to save the results
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark results
        """
        results = []
        
        print(f"Running {self.name} benchmark on {len(data)} questions...")
        
        for sample in tqdm(data):
                
            # Prepare input for the model
            input_data = self.prepare_input(sample)
            
            # Make prediction
            try:
                prediction = model.predict(
                    input_data, 
                    system_prompt=self.system_prompt
                )
                
                # Evaluate prediction
                evaluation = self.evaluate_prediction(sample, prediction)
                results.append({
                    "sample_id": sample.get("id", ""),
                    "input": input_data,
                    "prediction": prediction,
                    "evaluation": evaluation
                })
                
            except Exception as e:
                print(f"Error for question {sample.get('id', '')}: {e}")
                # Add failed sample to results
                results.append({
                    "sample_id": sample.get("id", ""),
                    "input": input_data,
                    "prediction": str(e),
                    "evaluation": {
                        "id": sample.get("id", ""),
                        "true_answer": sample.get("answer", ""),
                        "pred_answer": None,
                        "error": str(e)
                    }
                })        
        
        # Calculate aggregate metrics
        correct_count = sum(1 for result in results if result["evaluation"].get("correct", False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        metrics = {
            "num_samples": total_count,
            "num_correct": correct_count,
            "accuracy": accuracy
        }
        
        # Print summary metrics
        print(f"\n{self.name} Results:")
        print(f"Number of Questions: {metrics.get('num_samples', 0)}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        
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
#!/usr/bin/env python3
"""
Runner script for the clinical trial eligibility criteria design QA benchmark.
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import config
from benchmark import data
from benchmark.models import AzureOpenAIModel
from benchmark.tasks.qa_task import QATask

def run_benchmark(args):
    """
    Run the eligibility criteria design QA benchmark with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create configurations from arguments
    model_config = config.ModelConfig.from_args(args)
    task_config = config.TaskConfig.from_args(args)
    azure_config = config.AzureOpenAIConfig.from_args(args)
    
    # Create model
    model = AzureOpenAIModel.from_config(model_config, azure_config)
    
    # Set task name
    task_name = "eligibility_criteria_design"
    
    # Load dataset
    train_dataset, dataset = data.load_qa_task_data(task_name, task_config.data_path)
    
    if not dataset:
        raise ValueError(f"No data found for eligibility criteria design task")
    
    print(f"Loaded {len(train_dataset)} train samples and {len(dataset)} test samples for eligibility criteria design task")
    
    # Sample data if requested
    if task_config.num_samples is not None and task_config.num_samples > 0:
        dataset = data.sample_data(dataset, task_config.num_samples)
        print(f"Sampled {len(dataset)} test samples")
    
    # Create task
    task = QATask(task_name, system_prompt=args.system_prompt)
    
    # Set output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # add the sample size used for the benchmark in the filename
    filename = f"{timestamp}_n{task_config.num_samples}"
    output_dir = os.path.join(task_config.output_path, task_name, model_config.model_name, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "results.json")
    
    # Run benchmark
    print(f"Running eligibility criteria design benchmark with model {model_config.model_name}...")
    results = task.run_benchmark(model, dataset, output_path=output_path)
    
    # Print summary
    metrics = results["metrics"]
    print("\nBenchmark complete!")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Results saved to {output_path}")
    
    return results

def main():
    """Main entry point."""
    # Create argument parser
    parser = config.create_arg_parser("eligibility_criteria_design")
    
    # Add task-specific arguments
    parser.add_argument("--system-prompt", type=str, default=None,
                       help="Custom system prompt for the model")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(args)

if __name__ == "__main__":
    main() 
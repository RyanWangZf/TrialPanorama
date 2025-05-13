#!/usr/bin/env python3
"""
Runner script for the trial completion assessment benchmark.
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
from benchmark.tasks import TrialCompletionAssessmentTask

def run_benchmark(args):
    """
    Run the trial completion assessment benchmark with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create configurations from arguments
    model_config = config.ModelConfig.from_args(args)
    task_config = config.TaskConfig.from_args(args)
    azure_config = config.AzureOpenAIConfig.from_args(args)

    task_name = "trial_completion"

    # Create model
    model = AzureOpenAIModel.from_config(model_config, azure_config)
    
    # Load dataset
    _, test_data = data.load_trial_completion_data(task_config.data_path)
    
    if not test_data:
        raise ValueError(f"No test data found at {task_config.data_path}")
    
    print(f"Loaded {len(test_data)} test samples from {task_config.data_path}")
    
    # Sample data if requested
    if task_config.num_samples is not None and task_config.num_samples > 0:
        test_data = data.sample_data(test_data, task_config.num_samples)
        print(f"Sampled {len(test_data)} test samples")
    
    # Create task
    task = TrialCompletionAssessmentTask(system_prompt=args.system_prompt)
    
    # Set output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_n{task_config.num_samples}"
    output_dir = os.path.join(task_config.output_path, task_name, model_config.model_name, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "results.json")
    
    # Run benchmark
    print(f"Running benchmark with model {model_config.model_name}...")
    results = task.run_benchmark(model, test_data, output_path=output_path)
    
    # Print summary
    metrics = results["metrics"]
    print("\nBenchmark complete!")
    
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                if not isinstance(sub_v, dict):
                    print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
    
    print(f"Results saved to {output_path}")
    
    return results

def main():
    """Main entry point."""
    # Create argument parser
    parser = config.create_arg_parser("trial_completion")
    
    # Add task-specific arguments
    parser.add_argument("--system-prompt", type=str, default=None,
                       help="Custom system prompt for the model")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(args)

if __name__ == "__main__":
    main() 
"""
Configuration management for the benchmark framework.
"""

import os
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Default paths
DEFAULT_BENCHMARK_DATA_PATH = os.getenv("BENCHMARK_DATA_PATH")

# Task-specific data paths
STUDY_SEARCH_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "study_search")
STUDY_SCREENING_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "study_screening")
TRIAL_COMPLETION_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "trial_completion_assessment")
ARM_DESIGN_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "arm_design")
ELIGIBILITY_CRITERIA_DESIGN_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "eligibility_criteria_design")
ENDPOINT_DESIGN_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "endpoint_design")
EVIDENCE_SUMMARY_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "evidence_summary")
SAMPLE_SIZE_ESTIMATION_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_DATA_PATH, "sample_size_estimation")

# Output paths
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")

@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    reasoning_effort: str = "low"
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort
        )

@dataclass
class TaskConfig:
    """Configuration for the task."""
    data_path: str
    output_path: str
    task_name: str
    num_samples: int = None
    
    @classmethod
    def from_args(cls, args):
        return cls(
            data_path=args.data_path,
            output_path=args.output_path,
            task_name=args.task_name,
            num_samples=args.num_samples
        )

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI."""
    api_key: str = None
    api_base: str = None
    api_version: str = "2024-12-01-preview"
    
    def __post_init__(self):
        # Always prioritize environment variables from .env
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key or not self.api_base:
            print("Warning: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not found in environment.")
            print("Make sure you have a .env file in the directory with these variables.")
    
    @classmethod
    def from_args(cls, args):
        # We'll still return an instance, but the __post_init__ will prioritize env vars
        return cls(
            api_version=args.azure_api_version
        )

@dataclass
class PubMedConfig:
    """Configuration for PubMed API."""
    email: str = None
    api_key: str = None
    tool: str = "clinical_trial_crawler"
    
    def __post_init__(self):
        # Try to get from environment variables if not provided
        if self.email is None:
            self.email = os.environ.get("PUBMED_EMAIL")
        if self.api_key is None:
            self.api_key = os.environ.get("PUBMED_API_KEY")
    
    @classmethod
    def from_args(cls, args):
        return cls(
            email=args.pubmed_email,
            api_key=args.pubmed_api_key,
            tool=args.pubmed_tool
        )

def add_model_args(parser):
    """Add model-related arguments to an argument parser."""
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model sampling")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--reasoning-effort", type=str, default="low", help="Reasoning effort for reasoning models")

def add_task_args(parser, task_name, default_data_path=None):
    """Add task-related arguments to an argument parser."""
    if default_data_path is None:
        if task_name == "study_search":
            default_data_path = STUDY_SEARCH_DATA_PATH
        elif task_name == "study_screening":
            default_data_path = STUDY_SCREENING_DATA_PATH
        elif task_name == "trial_completion":
            default_data_path = TRIAL_COMPLETION_DATA_PATH
        elif task_name == "arm_design":
            default_data_path = ARM_DESIGN_DATA_PATH
        elif task_name == "eligibility_criteria_design":
            default_data_path = ELIGIBILITY_CRITERIA_DESIGN_DATA_PATH
        elif task_name == "endpoint_design":
            default_data_path = ENDPOINT_DESIGN_DATA_PATH
        elif task_name == "evidence_summary":
            default_data_path = EVIDENCE_SUMMARY_DATA_PATH
        elif task_name == "sample_size_estimation":
            default_data_path = SAMPLE_SIZE_ESTIMATION_DATA_PATH
        else:
            raise ValueError(f"Invalid task name: {task_name}")
    
    parser.add_argument("--data-path", type=str, default=default_data_path, help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH, 
                        help="Path to save the results")
    parser.add_argument("--task-name", type=str, default=task_name, help="Name of the task")
    parser.add_argument("--num-samples", type=int, default=None, 
                        help="Number of samples to evaluate (if None, use all)")

def add_azure_openai_args(parser):
    """Add Azure OpenAI-related arguments to an argument parser."""
    parser.add_argument("--azure-api-version", type=str, default="2024-12-01-preview", 
                        help="Azure OpenAI API version")

def add_pubmed_args(parser):
    """Add PubMed-related arguments to an argument parser."""
    parser.add_argument("--pubmed-email", type=str, default=None, help="Email for PubMed API")
    parser.add_argument("--pubmed-api-key", type=str, default=None, help="PubMed API key")
    parser.add_argument("--pubmed-tool", type=str, default="clinical_trial_crawler", 
                        help="Tool name for PubMed API")

def create_arg_parser(task_name):
    """Create an argument parser for the given task."""
    parser = argparse.ArgumentParser(description=f"Run the {task_name} benchmark.")
    
    # Core arguments
    add_model_args(parser)
    add_task_args(parser, task_name)
    
    # API-specific arguments
    add_azure_openai_args(parser)
    add_pubmed_args(parser)
    
    return parser 
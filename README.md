# TrialPanorama:Clinical Trial Database and Benchmark

This benchmark framework provides tools for evaluating language models on clinical trial-related tasks, designed to assess model capabilities in understanding and reasoning about medical research:

1. **Systematic Review Tasks**:
   - **Study Search**: Finding relevant clinical trials given a systematic review setup
   - **Study Screening**: Determining whether clinical trials should be included in a systematic review based on eligibility criteria
   - **Evidence Summary**: Generating evidence summaries from clinical trial data
3. **Clinical Trial Design Tasks**:
   - **Trial Completion Assessment**: Predicting whether a clinical trial will complete successfully or terminate prematurely, including the reason for termination
   - **Arm Design**: Designing appropriate trial arms
   - **Eligibility Criteria Design**: Creating inclusion/exclusion criteria
   - **Endpoint Design**: Defining primary and secondary outcomes
   - **Sample Size Estimation**: Calculating appropriate sample sizes

## Data

- TrialPanorama-database: https://huggingface.co/datasets/zifeng-ai/TrialPanorama-database

- TrialPanorama-benchmark: https://huggingface.co/datasets/zifeng-ai/TrialPanorama-benchmark

## Installation

The benchmark framework requires Python 3.8+ and uses pipenv for dependency management.

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install dependencies using pipenv
pipenv install

# Activate the virtual environment
pipenv shell
```

All benchmark scripts should be run within the pipenv virtual environment.

## Setup Environment Variables

Create a `.env` file in the root directory with your API keys:

```
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint

# Azure OpenAI Deployment Configuration
AZURE_DEPLOYMENT_GPT4O=deployment_name_for_gpt4o
AZURE_DEPLOYMENT_GPT4O_MINI=deployment_name_for_gpt4o_mini
AZURE_DEPLOYMENT_O3_MINI=deployment_name_for_o3_mini

# PubMed API (optional, for study search)
PUBMED_API_KEY=your_pubmed_api_key

# Path Configuration
DATABASE_PATH=/path/to/benchmark_data
BENCHMARK_DATA_PATH=/path/to/benchmark_results
```

The `.env` file will be automatically loaded by pipenv when you activate the virtual environment.

## Dataset Structure

Benchmark datasets should be structured as follows:

```
/path/to/benchmark_data/
├── study_search/
│   ├── train.jsonl
│   └── test.jsonl
├── study_screening/
│   ├── train.jsonl
│   └── test.jsonl
├── trial_completion/
│   ├── train.jsonl
│   └── test.jsonl
├── design_arms_qa/
│   ├── train.jsonl
│   └── test.jsonl
├── design_criteria_qa/
│   ├── train.jsonl
│   └── test.jsonl
├── design_outcome_qa/
│   ├── train.jsonl
│   └── test.jsonl
├── evidence_summary_qa/
│   ├── train.jsonl
│   └── test.jsonl
└── sample_size_estimation/
    ├── train.jsonl
    └── test.jsonl
```

Each dataset follows a specific format detailed in the task documentation.

## Verifying System Setup

Before running full benchmarks, you can verify your setup using the sanity check script:

```bash
# Make sure you're in the pipenv shell
pipenv shell

# Run with default settings (gpt-4o-mini and 2 samples per task)
./sanity_check.sh
```

This script:
- Runs each benchmark with a minimal number of samples
- Reports success/failure for each task
- Logs detailed output for troubleshooting
- Creates a `logs/` directory with results from each test

If all tests pass, your environment is correctly configured for running benchmarks.

## Running Individual Benchmarks

Ensure you're in the pipenv environment before running any benchmark script:

```bash
pipenv shell
```

### Study Search

```bash
# Run a single model
python benchmark_scripts/run_study_search.py --model-name gpt-4o

# Customize the system prompt
python benchmark_scripts/run_study_search.py --model-name gpt-4o --system-prompt "Your custom system prompt"

# Limit the number of samples
python benchmark_scripts/run_study_search.py --model-name gpt-4o --num-samples 10
```

### Study Screening

```bash
# Run a single model
python benchmark_scripts/run_study_screening.py --model-name gpt-4o

# Customize options
python benchmark_scripts/run_study_screening.py --model-name gpt-4o --system-prompt "Custom prompt" --num-samples 10
```

### Trial Completion Assessment

```bash
# Run a single model
python benchmark_scripts/run_trial_completion.py --model-name gpt-4o

# Customize options
python benchmark_scripts/run_trial_completion.py --model-name gpt-4o --system-prompt "Custom prompt" --num-samples 10
```

### Clinical Trial Design Tasks

```bash
# Arm Design
python benchmark_scripts/run_arm_design.py --model-name gpt-4o

# Eligibility Criteria Design
python benchmark_scripts/run_eligibility_criteria_design.py --model-name gpt-4o

# Endpoint Design
python benchmark_scripts/run_endpoint_design.py --model-name gpt-4o

# Evidence Summary
python benchmark_scripts/run_evidence_summary.py --model-name gpt-4o

# Sample Size Estimation
python benchmark_scripts/run_sample_size_estimation.py --model-name gpt-4o
```

## Running Batch Benchmarks

For each task, a shell script is provided to run benchmarks for multiple models in parallel. Make sure you're in the pipenv environment before running these scripts:

```bash
pipenv shell
```

Then run the desired benchmark script:

```bash
# Study-related Tasks
./benchmark_study_search.sh
./benchmark_study_screening.sh
./benchmark_evidence_summary.sh

# Design-related Tasks
./benchmark_arm_design.sh
./benchmark_eligibility_criteria_design.sh
./benchmark_endpoint_design.sh
./benchmark_evidence_summary.sh
./benchmark_sample_size_estimation.sh
./benchmark_trial_completion.sh
```

These scripts:
- Run benchmarks for three models (`gpt-4o-mini`, `gpt-4o`, and `o3-mini`)
- Use `nohup` to ensure processes continue even if your terminal session closes
- Log output to task-specific log files
- Return process IDs for monitoring

### Checking Batch Progress

```bash
# View running benchmark processes
ps aux | grep run_

# Check log files for a specific task
tail -f logs/endpoint_design_gpt-4o.log
```

## Benchmark Output Structure

Results are saved to the path specified in your `.env` file (BENCHMARK_DATA_PATH):

```
benchmark_results/
├── study_search/
│   └── [model_name]/
│       └── [timestamp]/
│           ├── results.json       # Detailed results including predictions
│           └── metrics.json       # Summary metrics only
├── study_screening/
├── trial_completion/
├── design_arms_qa/
├── design_criteria_qa/
├── design_outcome_qa/
├── evidence_summary_qa/
└── sample_size_estimation/
```

## Performance Metrics

Each task reports specific metrics:

### Study Search & Screening Metrics

- `precision`, `recall`, `f1`: Standard classification metrics
- `accuracy`: Overall accuracy across all studies

### Trial Completion Metrics

- `outcome_prediction.accuracy`: Accuracy of predicting completion vs termination
- `termination_type.accuracy`: Accuracy of predicting the specific termination reason

### QA Tasks Metrics

- `accuracy`: Proportion of questions answered correctly
- `f1_score`: For questions with potentially multiple correct answers

## Customizing the Framework

### Models

The framework currently supports Azure OpenAI models (`gpt-4o`, `gpt-4o-mini`, `o3-mini`, etc.).

You can modify `benchmark/models/` to add support for additional models.

### Tasks

Each task is implemented as a class that inherits from the base `Task` class.

Task classes handle:
- Input preparation
- Prediction parsing
- Evaluation against ground truth
- Results calculation


# Reference

If you find this project useful, please cite the following paper:

```bibtex
@article{wang2025trialpanorama,
  title     = {TrialPanorama: Database and Benchmark for Systematic Review and Design of Clinical Trials},
  author    = {Wang, Zifeng and Jin, Qiao and Lin, Jiacheng and Gao, Junyi and Pradeepkumar, Jathurshan and Jiang, Pengcheng and Danek, Benjamin and Lu, Zhiyong and Sun, Jimeng},
  year      = {2025},
}
```


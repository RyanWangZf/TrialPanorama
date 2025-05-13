#!/bin/bash
# Sequential benchmark script for trial completion assessment models


# Exit on error
set -e

echo "===== Starting trial completion benchmark sequence ====="

echo "Running benchmark with gpt-4o-mini..."
python run_trial_completion.py --model-name gpt-4o-mini > logs/trial_completion_gpt-4o-mini.log 2>&1 && \
echo "✓ gpt-4o-mini benchmark completed successfully"

echo "Running benchmark with gpt-4o..."
python run_trial_completion.py --model-name gpt-4o > logs/trial_completion_gpt-4o.log 2>&1 && \
echo "✓ gpt-4o benchmark completed successfully"

echo "Running benchmark with o3-mini..."
python run_trial_completion.py --model-name o3-mini > logs/trial_completion_o3-mini.log 2>&1 && \
echo "✓ o3-mini benchmark completed successfully"

echo "===== All trial completion benchmarks completed successfully =====" 



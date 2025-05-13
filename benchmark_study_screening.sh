#!/bin/bash
# Sequential benchmark script for study screening models

# Exit on error
set -e

echo "===== Starting screening benchmark sequence ====="

echo "Running benchmark with gpt-4o-mini..."
python run_study_screening.py --model-name gpt-4o-mini > logs/study_screening_gpt-4o-mini.log 2>&1 && \
echo "✓ gpt-4o-mini benchmark completed successfully"

echo "Running benchmark with gpt-4o..."
python run_study_screening.py --model-name gpt-4o > logs/study_screening_gpt-4o.log 2>&1 && \
echo "✓ gpt-4o benchmark completed successfully"

echo "Running benchmark with o3-mini..."
python run_study_screening.py --model-name o3-mini > logs/study_screening_o3-mini.log 2>&1 && \
echo "✓ o3-mini benchmark completed successfully"

echo "===== All screening benchmarks completed successfully ====="
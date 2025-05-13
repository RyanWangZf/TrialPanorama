#!/bin/bash
# Sequential benchmark script for study search models

# Exit on error
set -e

echo "===== Starting benchmark sequence ====="

echo "Running benchmark with gpt-4o-mini..."
python run_study_search.py --model-name gpt-4o-mini > logs/study_search_gpt-4o-mini.log 2>&1 && \
echo "✓ gpt-4o-mini benchmark completed successfully"

echo "Running benchmark with gpt-4o..."
python run_study_search.py --model-name gpt-4o > logs/study_search_gpt-4o.log 2>&1 && \
echo "✓ gpt-4o benchmark completed successfully"

echo "Running benchmark with o3-mini..."
python run_study_search.py --model-name o3-mini > logs/study_search_o3-mini.log 2>&1 && \
echo "✓ o3-mini benchmark completed successfully"

echo "===== All benchmarks completed successfully ====="
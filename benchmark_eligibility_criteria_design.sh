#!/bin/bash
# Sequential benchmark script for eligibility criteria design QA models

# Exit on error
set -e

echo "===== Starting eligibility criteria design benchmark sequence ====="

echo "Starting benchmark with gpt-4o-mini..."
nohup python -u run_eligibility_criteria_design.py --model-name gpt-4o-mini > logs/eligibility_criteria_gpt-4o-mini.log 2>&1 &
PID_MINI=$!
echo "✓ gpt-4o-mini benchmark started with PID: $PID_MINI"

echo "Starting benchmark with gpt-4o..."
nohup python -u run_eligibility_criteria_design.py --model-name gpt-4o > logs/eligibility_criteria_gpt-4o.log 2>&1 &
PID_4O=$!
echo "✓ gpt-4o benchmark started with PID: $PID_4O"

echo "Starting benchmark with o3-mini..."
nohup python -u run_eligibility_criteria_design.py --model-name o3-mini > logs/eligibility_criteria_o3-mini.log 2>&1 &
PID_O3_MINI=$!
echo "✓ o3-mini benchmark started with PID: $PID_O3_MINI"

echo "===== All eligibility criteria design benchmarks have been started ====="
echo "Check individual log files for progress:"
echo "  - eligibility_criteria_gpt-4o-mini.log"
echo "  - eligibility_criteria_gpt-4o.log"
echo "  - eligibility_criteria_o3-mini.log" 
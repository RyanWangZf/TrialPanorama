#!/bin/bash
# Sequential benchmark script for evidence summary QA models

# Exit on error
set -e

echo "===== Starting evidence summary benchmark sequence ====="

echo "Starting benchmark with gpt-4o-mini..."
nohup python -u run_evidence_summary.py --model-name gpt-4o-mini > logs/evidence_summary_gpt-4o-mini.log 2>&1 &
PID_MINI=$!
echo "✓ gpt-4o-mini benchmark started with PID: $PID_MINI"

echo "Starting benchmark with gpt-4o..."
nohup python -u run_evidence_summary.py --model-name gpt-4o > logs/evidence_summary_gpt-4o.log 2>&1 &
PID_4O=$!
echo "✓ gpt-4o benchmark started with PID: $PID_4O"

echo "Starting benchmark with o3-mini..."
nohup python -u run_evidence_summary.py --model-name o3-mini > logs/evidence_summary_o3-mini.log 2>&1 &
PID_O3_MINI=$!
echo "✓ o3-mini benchmark started with PID: $PID_O3_MINI"

echo "===== All evidence summary benchmarks have been started ====="
echo "Check individual log files for progress:"
echo "  - evidence_summary_gpt-4o-mini.log"
echo "  - evidence_summary_gpt-4o.log"
echo "  - evidence_summary_o3-mini.log" 
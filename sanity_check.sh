#!/bin/bash
# Sanity check script to verify benchmark scripts are working correctly
# Usage: ./sanity_check.sh [--model-name MODEL] [--num-samples N]

# Default values
MODEL=${1:-"gpt-4o-mini"}
NUM_SAMPLES=${2:-2}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "===== Running benchmark sanity checks with model: $MODEL and $NUM_SAMPLES samples ====="

# Function to run test and check result
run_test() {
    local script=$1
    local name=$2
    
    echo -n "Testing $name... "
    output=$(python benchmark_scripts/$script --model-name $MODEL --num-samples $NUM_SAMPLES 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "✅ PASSED"
        echo "$output" > logs/sanity_${name}.log
    else
        echo "❌ FAILED"
        echo "$output" > logs/sanity_${name}.log
        echo "Check logs/sanity_${name}.log for details"
    fi
}

# Run tests for each benchmark script
run_test "run_arm_design.py" "arm_design"
run_test "run_eligibility_criteria_design.py" "eligibility_criteria"
run_test "run_endpoint_design.py" "endpoint_design"
run_test "run_evidence_summary.py" "evidence_summary"
run_test "run_sample_size_estimation.py" "sample_size_estimation"
run_test "run_trial_completion.py" "trial_completion"
run_test "run_study_search.py" "study_search"
run_test "run_study_screening.py" "study_screening"

echo ""
echo "===== Sanity check complete ====="
echo "All logs saved to the logs/ directory"

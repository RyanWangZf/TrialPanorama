"""
Visualization script for the study search benchmark.
"""
import pdb
import os
import json
import pandas as pd
from glob import glob
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")
BENCHMARK_DATA_PATH = os.path.join(os.getenv("DATABASE_PATH"), "benchmark_data/final_results/trial_completion_assessment")


def trial_completion_data_eda():
    """
    Perform exploratory data analysis on the trial completion data.
    """
    stats_file = os.path.join(BENCHMARK_DATA_PATH, "stats.json")
    with open(stats_file, "r") as f:
        stats = json.load(f)

    print("Total number of train studies complete:", stats["training_set"]["outcome_distribution"]["complete"])
    print("Total number of train studies terminate:", stats["training_set"]["outcome_distribution"]["terminate"])
    
    print("Total number of test studies complete:", stats["test_set"]["outcome_distribution"]["complete"])
    print("Total number of test studies terminate:", stats["test_set"]["outcome_distribution"]["terminate"])

def load_trial_completion_results():
    """Load the trial completion results."""
    all_models = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
    all_results = []
    for model in all_models:
        result_folders = glob(os.path.join(DEFAULT_OUTPUT_PATH, "trial_completion", f"{model}/*"))
        for result_folder in result_folders:
            if not os.path.isdir(result_folder):
                continue

            result_file = os.path.join(result_folder, "results.json")
            if not os.path.exists(result_file):
                continue
            with open(result_file, "r") as f:
                results = json.load(f)
            this_result = {}
            this_result["model"] = model
            this_result["timestamp"] = os.path.basename(result_folder)
            this_result["num_samples"] = results["metrics"]["num_samples"]
            this_result["outcome_prediction_accuracy"] = results["metrics"]["outcome_prediction"]["accuracy"]
            this_result["outcome_prediction_precision"] = results["metrics"]["outcome_prediction"]["precision"]
            this_result["outcome_prediction_recall"] = results["metrics"]["outcome_prediction"]["recall"]
            this_result["outcome_prediction_f1_score"] = results["metrics"]["outcome_prediction"]["f1"]
            this_result["termination_type_prediction_accuracy"] = results["metrics"]["termination_type"]["accuracy"]
            all_results.append(this_result)
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(DEFAULT_OUTPUT_PATH, "trial_completion", "all_results.csv"), index=False)

if __name__ == "__main__":
    # load the results
    trial_completion_data_eda()
    load_trial_completion_results()

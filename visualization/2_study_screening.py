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
BENCHMARK_DATA_PATH = os.path.join(os.getenv("DATABASE_PATH"), "benchmark_data/final_results/study_screening")

def study_screening_data_eda():
    """
    Perform exploratory data analysis on the study screening data.
    """
    stats_file = os.path.join(BENCHMARK_DATA_PATH, "stats.json")
    with open(stats_file, "r") as f:
        stats = json.load(f)
    print("Total number of train reviews:", len(stats["train"]))
    print("Total number of test reviews:", len(stats["test"]))

    num_test_included = 0
    num_test_excluded = 0
    for test_data in stats["test"]:
        num_test_included += test_data.get('total_included', 0)
        num_test_excluded += test_data.get('total_excluded', 0)
    print("Total number of test studies included:", num_test_included)
    print("Total number of test studies excluded:", num_test_excluded)

    num_train_included = 0
    num_train_excluded = 0
    for train_data in stats["train"]:
        num_train_included += train_data.get('total_included', 0)
        num_train_excluded += train_data.get('total_excluded', 0)
    print("Total number of train studies included:", num_train_included)
    print("Total number of train studies excluded:", num_train_excluded)

def load_study_screening_results():
    """Load the study screening results."""
    all_models = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
    all_results = []
    for model in all_models:
        result_folders = glob(os.path.join(DEFAULT_OUTPUT_PATH, "study_screening", f"{model}/*"))
        for result_folder in result_folders:
            if not os.path.isdir(result_folder):
                continue

            result_file = os.path.join(result_folder, "results.json")
            with open(result_file, "r") as f:
                results = json.load(f)
            this_result = {}
            this_result["model"] = model
            this_result["timestamp"] = os.path.basename(result_folder)
            this_result["num_samples"] = results["metrics"]["num_samples"]
            for k, v in results["metrics"].items():
                if k in ["recall", "precision", "f1_score"]:
                    this_result[k] = v
                else:
                    pass
            all_results.append(this_result)
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(DEFAULT_OUTPUT_PATH, "study_screening", "all_results.csv"), index=False)

if __name__ == "__main__":
    # load the results
    study_screening_data_eda()
    load_study_screening_results()

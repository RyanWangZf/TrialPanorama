import json
import pandas as pd
import os
from pathlib import Path

# Get the repository root directory (parent of data_pipeline)
REPO_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = REPO_ROOT / "TrialPanorama-benchmark"
OUTPUT_DIR = REPO_ROOT / "data_pipeline"

def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def create_instruction_prompt():
    """Create instruction prompt for study screening task."""
    return ("You are a systematic review expert specializing in study screening and selection. "
            "Based on the provided systematic review background, objectives, and selection criteria, "
            "please determine whether each study should be included or excluded from the review.")

def format_screening_question(item):
    """Format the screening question with background, objectives, selection criteria, and studies."""
    inputs = item.get('inputs', {})
    context = item.get('context', [])
    
    # Extract background information
    background = inputs.get('background', '')
    objectives = inputs.get('objectives', '')
    selection_criteria = inputs.get('selection criteria', '')
    
    # Format the question
    question = f"""Background: {background}

Objectives: {objectives}

Selection Criteria: {selection_criteria}

Please screen the following studies and determine whether each should be included or excluded based on the selection criteria:

"""
    
    # Add each study with its details
    for i, study in enumerate(context, 1):
        title = study.get('title', '')
        abstract = study.get('abstract', '')
        pmid = study.get('PMID', '')
        
        question += f"Study {i}:\n"
        question += f"PMID: {pmid}\n"
        question += f"Title: {title}\n"
        question += f"Abstract: {abstract}\n\n"
    
    return question

def format_screening_answer(item):
    """Format the answer as a dictionary mapping study numbers to inclusion/exclusion decisions."""
    labels = item.get('labels', [])
    
    answer_dict = {}
    for i, label_info in enumerate(labels, 1):
        decision = label_info.get('label', '')
        answer_dict[str(i)] = decision.upper()
    
    return str(answer_dict)

def build_screening_sft_data(base_path=None):
    """
    Build SFT (Supervised Fine-Tuning) data from study screening training JSONL file.
    
    Args:
        base_path: Path to the benchmark data directory. If None, uses REPO_ROOT/TrialPanorama-benchmark
    
    Returns:
        pandas.DataFrame: DataFrame with columns [id, task_type, instruction_prompt, question, answer]
    """
    if base_path is None:
        base_path = RAW_DATA_DIR
    
    all_data = []
    
    train_file = os.path.join(base_path, "study_screening", "train.jsonl")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return pd.DataFrame()
        
    print(f"Loading data from {train_file}...")
    task_data = load_jsonl_data(train_file)
    
    instruction_prompt = create_instruction_prompt()
    
    for idx, item in enumerate(task_data):
        # Format the question with all the screening information
        question_text = format_screening_question(item)
        
        # Format the answer with inclusion/exclusion decisions
        answer_text = format_screening_answer(item)
        
        sft_row = {
            'id': idx,
            'task_type': 'study_screening',
            'instruction_prompt': instruction_prompt,
            'question': question_text,
            'answer': answer_text
        }
        
        all_data.append(sft_row)
        
        # Print progress for large files
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} examples...")
    
    df = pd.DataFrame(all_data)
    print(f"Created SFT dataset with {len(df)} examples for study screening")
    
    return df

def save_screening_sft_data(df, output_dir=None):
    """
    Save the study screening SFT data to both CSV and Parquet formats.
    
    Args:
        output_dir: Directory to save the output files. If None, uses REPO_ROOT/data_pipeline
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sft_study_screening_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved study screening SFT data to {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_dir, "sft_study_screening_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved study screening SFT data to {parquet_path}")
    
    return csv_path, parquet_path

def main():
    """Main function to build and save study screening SFT data."""
    print("Building SFT training data for study screening...")
    
    # Build the SFT dataset
    sft_df = build_screening_sft_data()
    
    if sft_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Display sample data
    print("\nSample data:")
    print("Columns:", sft_df.columns.tolist())
    print("\nFirst example:")
    first_example = sft_df.iloc[0]
    for col in sft_df.columns:
        content = str(first_example[col])
        print(f"{col}: {content[:300]}{'...' if len(content) > 300 else ''}")
    
    print(f"\nDataFrame info:")
    print(f"Shape: {sft_df.shape}")
    print(f"Columns: {sft_df.columns.tolist()}")
    
    # Save the data
    csv_path, parquet_path = save_screening_sft_data(sft_df)
    
    print(f"\nStudy screening SFT data building complete!")
    print(f"Total examples: {len(sft_df)}")
    print(f"Files saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")

if __name__ == "__main__":
    main()

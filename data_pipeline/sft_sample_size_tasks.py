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
    """Create instruction prompt for sample size estimation task."""
    return ("You are a clinical trial design expert specializing in sample size estimation. "
            "Based on the clinical trial design and statistical assumptions provided, "
            "please estimate the appropriate sample size for the study.")

def build_sample_size_sft_data(base_path=None):
    """
    Build SFT (Supervised Fine-Tuning) data from sample size estimation training JSONL file.
    
    Args:
        base_path: Path to the benchmark data directory. If None, uses REPO_ROOT/TrialPanorama-benchmark
    
    Returns:
        pandas.DataFrame: DataFrame with columns [id, task_type, instruction_prompt, question, answer, reasoning_process]
    """
    if base_path is None:
        base_path = RAW_DATA_DIR
    
    all_data = []
    
    train_file = os.path.join(base_path, "sample_size_estimation", "train.jsonl")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return pd.DataFrame()
        
    print(f"Loading data from {train_file}...")
    task_data = load_jsonl_data(train_file)
    
    instruction_prompt = create_instruction_prompt()
    
    for idx, item in enumerate(task_data):
        # Extract the question text
        question_text = item.get('question', '')
        
        # Get the answer (sample size)
        answer = item.get('answer', '')
        
        # Get the reasoning process from explanation field
        reasoning_process = item.get('explanation', '')
        
        sft_row = {
            'id': idx,
            'task_type': 'sample_size_estimation',
            'instruction_prompt': instruction_prompt,
            'question': question_text,
            'answer': str(answer),
            'reasoning_process': reasoning_process
        }
        
        all_data.append(sft_row)
    
    df = pd.DataFrame(all_data)
    print(f"Created SFT dataset with {len(df)} examples for sample size estimation")
    
    return df

def save_sample_size_sft_data(df, output_dir=None):
    """
    Save the sample size SFT data to both CSV and Parquet formats.
    
    Args:
        output_dir: Directory to save the output files. If None, uses REPO_ROOT/data_pipeline
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sft_sample_size_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved sample size SFT data to {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_dir, "sft_sample_size_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved sample size SFT data to {parquet_path}")
    
    return csv_path, parquet_path

def main():
    """Main function to build and save sample size SFT data."""
    print("Building SFT training data for sample size estimation...")
    
    # Build the SFT dataset
    sft_df = build_sample_size_sft_data()
    
    if sft_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Display sample data
    print("\nSample data:")
    print("Columns:", sft_df.columns.tolist())
    print("\nFirst example:")
    first_example = sft_df.iloc[0]
    for col in sft_df.columns:
        print(f"{col}: {str(first_example[col])[:200]}{'...' if len(str(first_example[col])) > 200 else ''}")
    
    print(f"\nDataFrame info:")
    print(f"Shape: {sft_df.shape}")
    print(f"Columns: {sft_df.columns.tolist()}")
    
    # Save the data
    csv_path, parquet_path = save_sample_size_sft_data(sft_df)
    
    print(f"\nSample size SFT data building complete!")
    print(f"Total examples: {len(sft_df)}")
    print(f"Files saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")

if __name__ == "__main__":
    main()

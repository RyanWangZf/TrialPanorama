import json
import pandas as pd
import os
from pathlib import Path

# Get the repository root directory (parent of data_pipeline)
REPO_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = REPO_ROOT / "TrialPanorama-benchmark"
OUTPUT_DIR = REPO_ROOT / "data_pipeline"

design_tasks = [
    "arm_design",
    "eligibility_criteria_design", 
    "endpoint_design"
]

def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def create_instruction_prompt(task_type):
    """Create instruction prompts for different task types."""
    prompts = {
        "arm_design": "You are a clinical trial design expert. Please select the arm or intervention descriptions that belong to the clinical trial described below.",
        "eligibility_criteria_design": "You are a clinical trial design expert. Please select the eligibility criteria that correspond to the clinical trial described below.",
        "endpoint_design": "You are a clinical trial design expert. Please select the outcome measures that correspond to the clinical trial described below."
    }
    return prompts.get(task_type, "You are a clinical trial design expert.")

def build_sft_data(base_path=None):
    """
    Build SFT (Supervised Fine-Tuning) data from training JSONL files.
    
    Args:
        base_path: Path to the benchmark data directory. If None, uses REPO_ROOT/TrialPanorama-benchmark
    
    Returns:
        pandas.DataFrame: DataFrame with columns [id, task_type, instruction_prompt, question, answer]
    """
    if base_path is None:
        base_path = RAW_DATA_DIR
    
    all_data = []
    global_id = 0
    
    for task in design_tasks:
        train_file = os.path.join(base_path, task, "train.jsonl")
        
        if not os.path.exists(train_file):
            print(f"Warning: {train_file} not found, skipping...")
            continue
            
        print(f"Loading data from {train_file}...")
        task_data = load_jsonl_data(train_file)
        
        instruction_prompt = create_instruction_prompt(task)
        
        for item in task_data:
            # Extract the question text and format it properly
            question_text = item.get('question', '')
            
            # Get the correct answer (just the letter)
            correct_answer = item.get('answer', '')
            
            # If there are options, include them in the question for context
            options = item.get('options', {})
            if options and correct_answer in options:
                # Include the options in the question for context
                options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
                full_question = f"{question_text}\n\nOptions:\n{options_text}"
                # Answer is just the letter (A, B, C, or D)
                answer_text = correct_answer
            else:
                full_question = question_text
                answer_text = correct_answer
            
            sft_row = {
                'id': global_id,
                'task_type': task,
                'instruction_prompt': instruction_prompt,
                'question': full_question,
                'answer': answer_text
            }
            
            all_data.append(sft_row)
            global_id += 1
    
    df = pd.DataFrame(all_data)
    print(f"Created SFT dataset with {len(df)} examples across {len(design_tasks)} tasks")
    
    # Print summary statistics
    print("\nTask distribution:")
    print(df['task_type'].value_counts())
    
    return df

def save_sft_data(df, output_dir=None):
    """
    Save the SFT data to both CSV and Parquet formats.
    
    Args:
        output_dir: Directory to save the output files. If None, uses REPO_ROOT/data_pipeline
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sft_training_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved SFT data to {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_dir, "sft_training_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved SFT data to {parquet_path}")
    
    return csv_path, parquet_path

def main():
    """Main function to build and save SFT data."""
    print("Building SFT training data from TrialPanorama benchmark...")
    
    # Build the SFT dataset
    sft_df = build_sft_data()
    
    # Display sample data
    print("\nSample data:")
    print(sft_df.head(2))
    
    # Save the data
    csv_path, parquet_path = save_sft_data(sft_df)
    
    print(f"\nSFT data building complete!")
    print(f"Total examples: {len(sft_df)}")
    print(f"Files saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")

if __name__ == "__main__":
    main()

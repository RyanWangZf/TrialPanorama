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
    """Create instruction prompt for trial completion assessment task."""
    return ("You are a clinical trial expert specializing in trial completion assessment. "
            "Based on the provided clinical trial design information, including study characteristics, "
            "eligibility criteria, and arm design, please predict the trial completion outcome and "
            "termination type (if terminated).")

def format_completion_question(item):
    """Format the completion assessment question with trial design information."""
    inputs = item.get('inputs', {})
    context = item.get('context', [])
    
    # Extract study characteristics
    allocation = inputs.get('allocation', 'N/A')
    intervention_model = inputs.get('intervention_model', 'N/A')
    observational_model = inputs.get('observational_model', 'N/A')
    primary_purpose = inputs.get('primary_purpose', 'N/A')
    time_perspective = inputs.get('time_perspective', 'N/A')
    masking = inputs.get('masking', 'N/A')
    title = inputs.get('title', 'N/A')
    phase = inputs.get('phase', 'N/A')
    number_of_arms = inputs.get('number_of_arms', 'N/A')
    
    # Format the question
    question = f"""Please assess the completion status of the following clinical trial:

Trial Title: {title}

Study Design Characteristics:
- Allocation: {allocation}
- Intervention Model: {intervention_model}
- Observational Model: {observational_model}
- Primary Purpose: {primary_purpose}
- Time Perspective: {time_perspective}
- Masking: {masking}
- Phase: {phase}
- Number of Arms: {number_of_arms}

"""
    
    # Add context information (eligibility criteria and arm design)
    for ctx in context:
        ctx_type = ctx.get('type', '')
        
        if ctx_type == 'eligibility_criteria':
            criteria = ctx.get('criteria', '')
            gender = ctx.get('gender', '')
            min_age = ctx.get('minimum_age', '')
            max_age = ctx.get('maximum_age', '')
            healthy_volunteers = ctx.get('healthy_volunteers', '')
            
            question += f"""Eligibility Criteria:
{criteria}

- Gender: {gender}
- Minimum Age: {min_age}
- Maximum Age: {max_age}
- Healthy Volunteers: {healthy_volunteers}

"""
        
        elif ctx_type == 'arm_design':
            arms = ctx.get('arms', [])
            question += "Study Arms:\n"
            for arm in arms:
                group_type = arm.get('group_type', 'N/A')
                title_arm = arm.get('title', 'N/A')
                description = arm.get('description', 'N/A')
                question += f"- {title_arm} (Type: {group_type}): {description}\n"
            question += "\n"
    
    question += "Based on this trial design, predict the completion outcome and termination type (if applicable)."
    
    return question

def format_completion_answer(item):
    """Format the answer as a JSON dictionary with outcome and termination type."""
    labels = item.get('labels', {})
    
    outcome = labels.get('outcome', '')
    terminate_type = labels.get('terminate_type', '')
    
    # Create answer dictionary
    answer_dict = {
        "outcome": outcome,
        "termination_type": terminate_type if terminate_type else None
    }
    
    return json.dumps(answer_dict)

def build_completion_assessment_sft_data(base_path=None):
    """
    Build SFT (Supervised Fine-Tuning) data from trial completion assessment training JSONL file.
    
    Args:
        base_path: Path to the benchmark data directory. If None, uses REPO_ROOT/TrialPanorama-benchmark
    
    Returns:
        pandas.DataFrame: DataFrame with columns [id, task_type, instruction_prompt, question, answer]
    """
    if base_path is None:
        base_path = RAW_DATA_DIR
    
    all_data = []
    
    train_file = os.path.join(base_path, "trial_completion_assessment", "train.jsonl")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return pd.DataFrame()
        
    print(f"Loading data from {train_file}...")
    task_data = load_jsonl_data(train_file)
    
    instruction_prompt = create_instruction_prompt()
    
    for idx, item in enumerate(task_data):
        # Format the question with all the trial information
        question_text = format_completion_question(item)
        
        # Format the answer as JSON dictionary
        answer_text = format_completion_answer(item)
        
        sft_row = {
            'id': idx,
            'task_type': 'trial_completion_assessment',
            'instruction_prompt': instruction_prompt,
            'question': question_text,
            'answer': answer_text
        }
        
        all_data.append(sft_row)
        
        # Print progress for large files
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} examples...")
    
    df = pd.DataFrame(all_data)
    print(f"Created SFT dataset with {len(df)} examples for trial completion assessment")
    
    return df

def save_completion_assessment_sft_data(df, output_dir=None):
    """
    Save the trial completion assessment SFT data to both CSV and Parquet formats.
    
    Args:
        output_dir: Directory to save the output files. If None, uses REPO_ROOT/data_pipeline
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sft_trial_completion_assessment_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved trial completion assessment SFT data to {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_dir, "sft_trial_completion_assessment_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved trial completion assessment SFT data to {parquet_path}")
    
    return csv_path, parquet_path

def main():
    """Main function to build and save trial completion assessment SFT data."""
    print("Building SFT training data for trial completion assessment...")
    
    # Build the SFT dataset
    sft_df = build_completion_assessment_sft_data()
    
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
    csv_path, parquet_path = save_completion_assessment_sft_data(sft_df)
    
    print(f"\nTrial completion assessment SFT data building complete!")
    print(f"Total examples: {len(sft_df)}")
    print(f"Files saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")

if __name__ == "__main__":
    main()

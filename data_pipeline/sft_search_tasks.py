import json
import pandas as pd
import os
from pathlib import Path
import requests
from tqdm import tqdm
import time
import pdb

# Get the repository root directory (parent of data_pipeline)
REPO_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = REPO_ROOT / "TrialPanorama-benchmark"
OUTPUT_DIR = REPO_ROOT / "data_pipeline"

# vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8000/v1/completions"

def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def create_instruction_prompt():
    """Create instruction prompt for study search/retrieval task."""
    return """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.
Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[thinking process]
</think>
<answer>
{
    "query": "...."
} 
</answer>. 
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately. You don't need to rewrite the query when the query is already good.

Here's the user query:
    """

def format_review_information(item):
    """Format the retrieval question with systematic review information."""
    inputs = item.get('inputs', {})
    
    # Extract review information
    background = inputs.get('background', '')
    objectives = inputs.get('objectives', '')
    selection_criteria = inputs.get('selection criteria', '')
    
    # Format the question
    question = f"""Here is the background, objectives, and selection criteria of a systematic review:

Background: {background}

Objectives: {objectives}

Selection Criteria: {selection_criteria}
"""
    
    return question

def call_vllm_api(prompt, max_tokens=512, temperature=0.7, top_p=0.95):
    """Call the vLLM API to generate a response."""
    payload = {
        "model": "DeepRetrieval/DeepRetrieval-PubMed-3B-Llama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    try:
        response = requests.post(VLLM_SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['text'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM API: {e}")
        return None
    except Exception as e:
        print(f"Error processing response: {e}")
        return None

def build_retrieval_sft_data(base_path=None, max_examples=None):
    """
    Build SFT (Supervised Fine-Tuning) data from study search training JSONL file.
    Uses vLLM inference to generate answers.
    
    Args:
        base_path: Path to the benchmark data directory. If None, uses REPO_ROOT/TrialPanorama-benchmark
        max_examples: Maximum number of examples to process. If None, process all.
    
    Returns:
        pandas.DataFrame: DataFrame with columns [id, task_type, instruction_prompt, question, answer]
    """
    if base_path is None:
        base_path = RAW_DATA_DIR
    
    all_data = []
    
    train_file = os.path.join(base_path, "study_search", "train.jsonl")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return pd.DataFrame()
        
    print(f"Loading data from {train_file}...")
    task_data = load_jsonl_data(train_file)
    
    if max_examples is not None:
        task_data = task_data[:max_examples]
        print(f"Processing first {max_examples} examples...")
    
    instruction_prompt = create_instruction_prompt()
    
    print(f"\nRunning inference with vLLM server at {VLLM_SERVER_URL}...")
    print("Make sure the vLLM server is running (./vllm_host.sh)")
    
    # Test connection
    try:
        test_response = requests.get("http://localhost:8000/health", timeout=5)
        print("✓ vLLM server is reachable")
    except:
        print("⚠ Warning: Cannot reach vLLM server. Make sure it's running.")
        print("  Run: cd data_pipeline && ./vllm_host.sh")
        return pd.DataFrame()
    
    failed_count = 0
    
    for idx, item in enumerate(tqdm(task_data, desc="Processing examples")):
        # Format the question
        question_text = format_review_information(item)
        
        # Create the full prompt for the model
        full_prompt = f"{instruction_prompt}\n\nHere's the user query:{question_text}\n\nAssistant: Let me rewrite the query with reasoning. <think>"
        
        # Call vLLM API to generate answer
        answer_text = call_vllm_api(full_prompt)


        # print("--------------------------------")
        # print(item)
        print(answer_text)
        # print("--------------------------------")

        
        if answer_text is None:
            failed_count += 1
            # Use empty string as fallback
            answer_text = ""
            print(f"\nWarning: Failed to generate answer for example {idx}")
        
        sft_row = {
            'id': idx,
            'task_type': 'study_search',
            'instruction_prompt': instruction_prompt,
            'question': question_text,
            'answer': answer_text
        }
        
        all_data.append(sft_row)
        
        # Small delay to avoid overwhelming the server
        # time.sleep(0.1)
    
    df = pd.DataFrame(all_data)
    print(f"\nCreated SFT dataset with {len(df)} examples for study search")
    if failed_count > 0:
        print(f"⚠ Warning: {failed_count} examples failed to generate answers")
    
    return df

def save_retrieval_sft_data(df, output_dir=None):
    """
    Save the study search SFT data to both CSV and Parquet formats.
    
    Args:
        output_dir: Directory to save the output files. If None, uses REPO_ROOT/data_pipeline
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "sft_study_search_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved study search SFT data to {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_dir, "sft_study_search_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved study search SFT data to {parquet_path}")
    
    return csv_path, parquet_path

def run_deepretrieval_inference():
    """Main function to build and save study search SFT data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SFT data for study search using vLLM inference')
    parser.add_argument('--max-examples', type=int, default=None, 
                       help='Maximum number of examples to process (default: all)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature for vLLM (default: 0.7)')
    
    args = parser.parse_args()
    
    print("Building SFT study search data with vLLM inference...")
    print(f"vLLM Server: {VLLM_SERVER_URL}")
    
    # Build the SFT dataset with vLLM inference
    sft_df = build_retrieval_sft_data(max_examples=args.max_examples)
    
    if sft_df.empty:
        print("No data generated. Exiting.")
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
    csv_path, parquet_path = save_retrieval_sft_data(sft_df)
    
    print(f"\nStudy search SFT data building complete!")
    print(f"Total examples: {len(sft_df)}")
    print(f"Files saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")


def clean_data():
    """Clean the data and reformat instruction prompt and answer."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "sft_study_search_data.csv"))
    df = df.dropna(subset=["answer"])
    
    # Set the new instruction prompt
    new_instruction = "Your task is to create search query given the input question. The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."
    df['instruction_prompt'] = new_instruction
    
    # Extract query from the answer JSON
    def extract_query(answer_text):
        try:
            # Find the JSON content between <answer> and </answer> tags
            if '<answer>' in answer_text and '</answer>' in answer_text:
                start = answer_text.find('<answer>') + len('<answer>')
                end = answer_text.find('</answer>')
                json_str = answer_text[start:end].strip()
                answer_dict = json.loads(json_str)
                return answer_dict.get('query', '')
            else:
                # Try to parse the entire answer as JSON
                answer_dict = json.loads(answer_text)
                return answer_dict.get('query', '')
        except:
            # If parsing fails, return the original answer
            return answer_text
    
    df['answer'] = df['answer'].apply(extract_query)
    
    # Drop rows where answer is empty after extraction
    df = df[df['answer'].str.strip() != '']
    
    df.to_csv(os.path.join(OUTPUT_DIR, "sft_study_search_data_cleaned.csv"), index=False)
    df.to_parquet(os.path.join(OUTPUT_DIR, "sft_study_search_data_cleaned.parquet"), index=False)
    print(f"Cleaned data saved. Total rows: {len(df)}")
    
    return df


if __name__ == "__main__":
    run_deepretrieval_inference()
    clean_data()


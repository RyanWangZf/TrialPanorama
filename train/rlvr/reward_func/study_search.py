import re
import os
import time
import json
import requests
from collections import deque
from threading import Lock
from typing import Dict, List, Any, Optional
import random

# Global variables for rate limiting
_request_times = deque(maxlen=20)  # Track last 20 requests
_request_lock = Lock()  # Thread-safe lock for request tracking


def extract_solution(solution_str: str):
    """Extract the search query from the solution string."""
    # Remove everything before the first "Assistant:" if present
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip()
    else:
        # If no assistant header found, use the whole string
        processed_str = solution_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str


def validate_response_structure(processed_str: str, do_print: bool = False) -> bool:
    """Performs validation of response structure for study search.
    
    For study_search, we only require <answer> tags, NOT <think> tags.
    
    Args:
        processed_str: Processed response string from the model
        do_print: Whether to print validation details
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True
    
    # Check required tags (only answer tags for study_search)
    tags = {
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Check that <think> tags are NOT present (they shouldn't be for study_search)
    if '<think>' in processed_str or '</think>' in processed_str:
        if do_print:
            print("  [Error] <think> tags found but not expected for study_search task")
        validation_passed = False
    
    # Verify tag order (answer_start should come before answer_end)
    if positions.get('answer_start', -1) > positions.get('answer_end', -1):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed


def format_score(solution_str: str, format_reward: float = 0.1, do_print: bool = False) -> float:
    """Calculate format score for study search task.
    
    Args:
        solution_str: The generated solution string
        format_reward: Reward for correct format
        do_print: Whether to print debugging info
        
    Returns:
        Format score
    """
    if do_print:
        print("\n" + "="*80)
        print(" Format Score Calculation ".center(80, '='))
        print("="*80)
    
    # Extract answer and processed string
    answer_text, processed_str = extract_solution(solution_str)
    
    if do_print:
        print(f"\nExtracted answer: {answer_text}")
    
    # Validate structure
    format_correct = validate_response_structure(processed_str, do_print)
    
    # Calculate format score
    if format_correct:
        score = format_reward
        if do_print:
            if answer_text:
                print(f"\n✓ Format is correct, score: {score}")
            else:
                print(f"\n⚠ Format is correct but answer is empty, score: {score}")
    else:
        score = -2.0  # Penalty for incorrect format
        if do_print:
            print(f"\n✗ Format is incorrect, penalty: {score}")
    
    return score


def search_pubmed_with_retry(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 1000,
    max_retries: int = 3,
    do_print: bool = False
) -> List[str]:
    """Search PubMed with retry logic and rate limiting.
    
    Args:
        query: Search query string
        api_key: PubMed API key (optional)
        max_results: Maximum number of results to retrieve
        max_retries: Maximum number of retry attempts
        do_print: Whether to print debugging info
        
    Returns:
        List of PubMed IDs (as strings)
    """
    if not query or not query.strip():
        if do_print:
            print("[Warning] Empty query provided")
        return []
    
    # Rate limit checking
    current_time = time.time()
    with _request_lock:
        # Remove requests older than 1 second
        while _request_times and current_time - _request_times[0] > 1.0:
            _request_times.popleft()
        
        # Check if we're exceeding rate limit (10 requests per second)
        if len(_request_times) >= 10:
            if do_print:
                print("\033[93m[Warning] PubMed rate limit (10 req/s) reached! Consider reducing batch size.\033[0m")
            # Wait a bit to avoid rate limiting
            time.sleep(0.2)
        
        # Record this request
        _request_times.append(current_time)
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # Prepare parameters
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "usehistory": "y",
        "retmode": "json",
        "sort": "relevance",
    }
    
    if api_key:
        params["api_key"] = api_key
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            
            if do_print:
                print(f"[PubMed Search] Retrieved {len(ids)} results")
            
            return ids
            
        except requests.exceptions.Timeout:
            if do_print:
                print(f"[Warning] Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.exceptions.HTTPError as e:
            if do_print:
                print(f"[Warning] HTTP error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            if do_print:
                print(f"[Warning] Request error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        except json.JSONDecodeError as e:
            if do_print:
                print(f"[Warning] Failed to parse JSON response: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        except Exception as e:
            if do_print:
                print(f"[Warning] Unexpected error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    # If all retries failed
    if do_print:
        print(f"[Error] Failed to retrieve results after {max_retries} attempts")
    return []


def calculate_recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 100) -> float:
    """Calculate recall@k metric.
    
    Args:
        relevant_ids: List of relevant PubMed IDs
        retrieved_ids: List of retrieved PubMed IDs (in rank order)
        k: Number of top results to consider
        
    Returns:
        Recall@k score (between 0 and 1)
    """
    if not relevant_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    retrieved_at_k = set(retrieved_ids[:k])
    
    relevant_at_k = len(relevant_set & retrieved_at_k)
    recall = relevant_at_k / len(relevant_set)
    
    return recall


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    format_reward: float = 0.1,
    correct_reward: float = 1.0,
    data_source=None,
    extra_info=None
) -> float:
    """Compute the reward score for study search task.
    
    Args:
        solution_str: The generated solution string
        ground_truth: Dictionary containing:
            - relevant_pmids: List of relevant PubMed IDs
            - review_pmid: Review PubMed ID (for reference)
        format_reward: Reward for correct format (default: 0.1)
        correct_reward: Reward for correct answer (default: 1.0)
        do_print: Whether to print debugging info
        api_key: PubMed API key (optional, but recommended for rate limiting)
        
    Returns:
        Total reward score
    """


    
    do_print = random.randint(1, 32) == 1
    if do_print:
        print("\n" + "="*80)
        print(" Study Search Reward Computation ".center(80, '='))
        print("="*80)
    
    # Get API key from environment if not provided
    api_key = "35ec46736e2204273d637c839d33bd5b3508"
    
    # Extract ground truth
    relevant_pmids = ground_truth.get("relevant_pmids", [])
    
    if do_print:
        print(f"\nGround Truth:")
        print(f"  Number of relevant PMIDs: {len(relevant_pmids)}")
    solution_str = "<answer>" + solution_str

    if do_print:
        print(f"Solution string: {solution_str}")
    
    # Step 1: Calculate format score
    fmt_score = format_score(solution_str, format_reward, do_print)
    
    if fmt_score < 0:
        # Format is incorrect, return penalty
        if do_print:
            print("\n" + "-"*80)
            print(f" Final Score ".center(80, '-'))
            print(f"  Format: {fmt_score}")
            print(f"  Answer: 0.0 (skipped due to format error)")
            print(f"  Total: {fmt_score}")
            print("="*80 + "\n")
        return fmt_score
    
    # Step 2: Extract query and retrieve results
    query, _ = extract_solution(solution_str)
    
    if not query:
        if do_print:
            print("\n[Error] No query found in solution")
            print("\n" + "-"*80)
            print(f" Final Score ".center(80, '-'))
            print(f"  Format: {fmt_score}")
            print(f"  Answer: 0.0 (no query found)")
            print(f"  Total: 0.0")
            print("="*80 + "\n")
        return 0.0
    
    if do_print:
        print(f"\n[Query Extraction]")
        print(f"  Query: {query[:200]}{'...' if len(query) > 200 else ''}")
    
    # Step 3: Search PubMed
    try:
        retrieved_ids = search_pubmed_with_retry(
            query=query,
            api_key=api_key,
            max_results=1000,
            max_retries=3,
            do_print=do_print
        )
        
        if do_print:
            print(f"  Retrieved: {len(retrieved_ids)} documents")
        
    except Exception as e:
        if do_print:
            print(f"\n[Error] Failed to search PubMed: {e}")
        retrieved_ids = []
    
    # Step 4: Calculate recall@100
    answer_score = 0.0
    if retrieved_ids:
        recall_100 = calculate_recall_at_k(relevant_pmids, retrieved_ids, k=100)
        
        if do_print:
            print(f"\n[Answer Evaluation]")
            print(f"  Recall@100: {recall_100:.4f}")
        
        # Score is proportional to recall@100
        answer_score = recall_100 * correct_reward
        
        if do_print:
            print(f"  Answer score: {answer_score:.4f}")
    else:
        if do_print:
            print(f"\n[Warning] No results retrieved, answer score: 0.0")
    
    # Step 5: Calculate total score
    # Only add format score if answer score is positive
    if answer_score > 0:
        total_score = fmt_score + answer_score
    else:
        # If answer is wrong, no format bonus
        total_score = 0.0
    
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {fmt_score}")
        print(f"  Answer (Recall@100): {answer_score:.4f}")
        print(f"  Total: {total_score:.4f}")
        print("="*80 + "\n")
    
    return total_score


if __name__ == '__main__':
    # Test case 1: Correct format with a valid query
    solution_str1 = """<|start_header_id|>assistant<|end_header_id|>
<answer>
((Diabetes OR Type 2 Diabetes) AND (Metformin OR Antidiabetic agents) AND (Randomized Controlled Trial OR Clinical Trial))
</answer>"""
    
    ground_truth1 = {
        'relevant_pmids': ['12345678', '23456789', '34567890'],
        'review_pmid': '38078494'
    }
    
    print("Test 1 - Correct format with query:")
    score1 = compute_score(solution_str1, ground_truth1, format_reward=0.1, correct_reward=1.0)
    print(f"Score: {score1}\n")
    
    # Test case 2: Bad format (no answer tags)
    solution_str2 = """<|start_header_id|>assistant<|end_header_id|>
The search query should be: Diabetes AND Metformin"""
    
    ground_truth2 = {
        'relevant_pmids': ['12345678', '23456789'],
        'review_pmid': '38078494'
    }
    
    print("\nTest 2 - Bad format (no answer tags):")
    score2 = compute_score(solution_str2, ground_truth2, format_reward=0.1, correct_reward=1.0)
    print(f"Score: {score2}\n")
    assert score2 == -2.0, f"Expected score -2.0, got {score2}"
    
    # Test case 3: Correct format but empty query
    solution_str3 = """<|start_header_id|>assistant<|end_header_id|>
<answer>((Lamotrigine OR Lamotrigine Trial) AND (Epilepsy Treatment Trial OR Seizure Randomized controlled trial OR Antiepileptic drug Trial))</answer>"""
    
    ground_truth3 = {
        'relevant_pmids': ["20696552", "2612495", "8937535", "10563619", "2498073", "2127016", "8232944", "8112232", "17938371", "18077797", "8505632", "8453943"],
        'review_pmid': '38078494'
    }
    
    print("\nTest 3")
    score3 = compute_score(solution_str3, ground_truth3, format_reward=0.1, correct_reward=1.0)
    print(f"Score: {score3}\n")
    
    # Test case 4: Format with <think> tags (should fail for study_search)
    solution_str4 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Let me think about the search query...
</think>
<answer>
Diabetes AND Metformin
</answer>"""
    
    ground_truth4 = {
        'relevant_pmids': ['12345678'],
        'review_pmid': '38078494'
    }
    
    print("\nTest 4 - With <think> tags (should fail):")
    score4 = compute_score(solution_str4, ground_truth4, format_reward=0.1, correct_reward=1.0)
    print(f"Score: {score4}\n")
    assert score4 == -2.0, f"Expected score -2.0 for <think> tags, got {score4}"
    
    print("\n" + "="*80)
    print("Test suite completed!")
    print("="*80)


import re
import random
import json


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
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


def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        do_print: Whether to print validation details
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
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

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed


def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False
        
        data = json.loads(json_str)

        if not isinstance(data, dict):
            if do_print:
                print("Parsed data is not a JSON object:", data)
            return False
        
        # Required keys
        required_keys = {"answer"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False


def extract_number_from_answer(answer_text, do_print=False):
    """Extract a numeric value from the answer text (directly from <answer> tag content)."""
    if not answer_text:
        if do_print:
            print("[Error] Empty answer text")
        return None
    
    # Strip whitespace from answer
    answer_str = answer_text.strip()
    
    # Remove commas and other common formatting
    answer_str = answer_str.replace(',', '').replace(' ', '')
    
    # Try to extract a number
    number_match = re.search(r'(\d+(?:\.\d+)?)', answer_str)
    if number_match:
        return float(number_match.group(1))
    else:
        if do_print:
            print(f"[Error] Could not extract number from answer: {answer_text}")
        return None


def check_sample_size_correctness(pred_sample_size, true_sample_size, tolerance=0.2):
    """
    Check if the predicted sample size is within the tolerance range of the true value.
    
    Args:
        pred_sample_size: Predicted sample size (float or int)
        true_sample_size: True sample size (float or int)
        tolerance: Tolerance level (default 0.2 for 20%)
    
    Returns:
        Boolean indicating if prediction is correct
    """
    if pred_sample_size is None or true_sample_size is None:
        return False
    
    lower_bound = true_sample_size * (1 - tolerance)
    upper_bound = true_sample_size * (1 + tolerance)
    
    return lower_bound <= pred_sample_size <= upper_bound


def compute_score(solution_str, ground_truth, data_source=None, format_reward=0.1, correct_reward=1.0, extra_info=None):
    """The scoring function for sample size estimation task.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: dictionary containing 'target' (the true sample size)
        format_reward: the score for correct format but wrong answer (default: 0.1)
        correct_reward: the score for the correct answer (default: 1.0)
    
    Returns:
        float: total score
    """

    # add <think> at the beginning of the solution string if not already present
    # if not solution_str.strip().startswith("<think>"):
    # solution_str = "<think>" + solution_str
    
    # Extract target sample size from ground truth
    target = ground_truth.get('target')
    if target is None:
        print("[Error] No target found in ground_truth")
        return -2
    
    # Convert target to float/int
    try:
        true_sample_size = float(target)
    except (ValueError, TypeError):
        print(f"[Error] Could not convert target to number: {target}")
        return -2
    
    # Extract answer from solution string
    answer_text, processed_str = extract_solution(solution_str)
    
    # Random print for debugging (1 in 32 chance)
    do_print = random.randint(1, 32) == 1
    
    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    # json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct #and json_format_correct
    
    format_score = format_reward if format_correct else -2
    
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"answer_text: {answer_text}")
        print(f"Target sample size: {true_sample_size}")
        print(f"Format correct: {format_correct}")
    
    # Calculate answer score
    answer_score = 0
    if format_correct and answer_text:
        pred_sample_size = extract_number_from_answer(answer_text, do_print)
        
        if pred_sample_size is not None:
            is_correct = check_sample_size_correctness(pred_sample_size, true_sample_size, tolerance=0.2)
            
            if do_print:
                print(f"Predicted sample size: {pred_sample_size}")
                print(f"Correct: {is_correct}")
                print(f"Valid range: [{true_sample_size * 0.8:.1f}, {true_sample_size * 1.2:.1f}]")
            
            if is_correct:
                answer_score = correct_reward
    
    # Calculate total score
    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score
    
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score


if __name__ == '__main__':
    # Test case 1: Correct answer within 20% range
    solution_str1 = """<|start_header_id|>assistant<|end_header_id|>
<think>
To calculate the sample size, I need to consider the power, significance level, and effect size.
Based on the study design, a sample size of 500 would be appropriate.
</think>
<answer>500</answer>"""
    
    ground_truth1 = {'target': 511}
    score1 = compute_score(solution_str1, ground_truth1, format_reward=0.1, correct_reward=1.0)
    print(f"Test 1 - Correct answer (500 vs 511): Score = {score1}")
    assert score1 > 1.0, f"Expected score > 1.0, got {score1}"
    
    # Test case 2: Answer outside 20% range
    solution_str2 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Let me calculate...
</think>
<answer>200</answer>"""
    
    ground_truth2 = {'target': 511}
    score2 = compute_score(solution_str2, ground_truth2, format_reward=0.1, correct_reward=1.0)
    print(f"Test 2 - Wrong answer (200 vs 511): Score = {score2}")
    assert score2 == 0, f"Expected score 0, got {score2}"
    
    # Test case 3: Bad format
    solution_str3 = """<|start_header_id|>assistant<|end_header_id|>
The answer is 500 participants."""
    
    ground_truth3 = {'target': 511}
    score3 = compute_score(solution_str3, ground_truth3, format_reward=0.1, correct_reward=1.0)
    print(f"Test 3 - Bad format: Score = {score3}")
    assert score3 == -2, f"Expected score -2, got {score3}"
    
    # Test case 4: Good format but no valid number
    solution_str4 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Let me think about this...
</think>
<answer>Not enough information</answer>"""
    
    ground_truth4 = {'target': 511}
    score4 = compute_score(solution_str4, ground_truth4, format_reward=0.1, correct_reward=1.0)
    print(f"Test 4 - Good format, no valid number: Score = {score4}")
    assert score4 == 0, f"Expected score 0, got {score4}"
    
    # Test case 5: Edge case - exactly at 20% boundary (lower)
    solution_str5 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Calculating...
</think>
<answer>408.8</answer>"""
    
    ground_truth5 = {'target': 511}
    score5 = compute_score(solution_str5, ground_truth5, format_reward=0.1, correct_reward=1.0)
    print(f"Test 5 - Edge case lower bound (408.8 vs 511): Score = {score5}")
    assert score5 > 1.0, f"Expected score > 1.0, got {score5}"
    
    # Test case 6: Edge case - within upper boundary
    solution_str6 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Let me calculate the required sample size...
</think>
<answer>610</answer>"""
    
    ground_truth6 = {'target': 511}
    score6 = compute_score(solution_str6, ground_truth6, format_reward=0.1, correct_reward=1.0)
    print(f"Test 6 - Within upper boundary (610 vs 511, 20% range is {511*1.2:.1f}): Score = {score6}")
    assert score6 > 1.0, f"Expected score > 1.0, got {score6}"
    
    # Test case 7: Number with comma formatting
    solution_str7 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Sample size calculation...
</think>
<answer>1,500</answer>"""
    
    ground_truth7 = {'target': 1538}
    score7 = compute_score(solution_str7, ground_truth7, format_reward=0.1, correct_reward=1.0)
    print(f"Test 7 - Comma formatted number (1,500 vs 1,538): Score = {score7}")
    assert score7 > 1.0, f"Expected score > 1.0, got {score7}"
    
    # Test case 8: Answer just outside upper boundary
    solution_str8 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Calculating sample size...
</think>
<answer>650</answer>"""
    
    ground_truth8 = {'target': 511}
    score8 = compute_score(solution_str8, ground_truth8, format_reward=0.1, correct_reward=1.0)
    print(f"Test 8 - Just outside upper boundary (650 vs 511, upper limit is {511*1.2:.1f}): Score = {score8}")
    assert score8 == 0, f"Expected score 0, got {score8}"
    
    # Test case 9: Answer just outside lower boundary
    solution_str9 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Let me compute...
</think>
<answer>400</answer>"""
    
    ground_truth9 = {'target': 511}
    score9 = compute_score(solution_str9, ground_truth9, format_reward=0.1, correct_reward=1.0)
    print(f"Test 9 - Just outside lower boundary (400 vs 511, lower limit is {511*0.8:.1f}): Score = {score9}")
    assert score9 == 0, f"Expected score 0, got {score9}"
    
    # Test case 10: Exact match
    solution_str10 = """<|start_header_id|>assistant<|end_header_id|>
<think>
Based on the study design with 80% power and 5% significance level,
I calculate that we need exactly 511 participants.
</think>
<answer>511</answer>"""
    
    ground_truth10 = {'target': 511}
    score10 = compute_score(solution_str10, ground_truth10, format_reward=0.1, correct_reward=1.0)
    print(f"Test 10 - Exact match (511 vs 511): Score = {score10}")
    assert score10 > 1.0, f"Expected score > 1.0, got {score10}"
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


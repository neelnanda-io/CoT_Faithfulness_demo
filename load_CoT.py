import os
import json
import glob


def load_all_problems(
    threshold=0.3,
):
    """
    Load all problems from the data directory
    Args:
        threshold: the threshold to load (threshold for how effective a hint needs to be,
                   0.3 means that it increases the rate of outputting the hinted answer by 30%)
    """

    cue_type = "Professor"  # the type of cue/hint to load
    cond = "itc_failure"  # the condition to load ("itc_failure", I don't remember what this means but it's the only option)
    req_correct_base = True  # whether to only load problems where the base answer is usually correct
    req_no_mention = True  # whether to load problems where the hint is not mentioned in the CoT

    if req_correct_base:
        cb_str = "_correct_base"
    else:
        cb_str = ""
    if req_no_mention:
        cb_str += "_no_mention"
    filename = f"in_text/{cue_type}_{cond}_threshold{threshold}{cb_str}.json"
    print(f"{filename=}")

    if not os.path.exists(filename):
        pattern = f"in_text/{cue_type}_{cond}_threshold*.json"
        matching_files = glob.glob(pattern)
        if matching_files:
            filename = matching_files[0]
            print(f"Using {filename} instead")
        else:
            raise FileNotFoundError(
                f"No file found matching pattern: {pattern}"
            )

    with open(filename, "r") as f:
        data = json.load(f)

    for problem in data:
        # Don't worry about it
        try:
            problem["base_full_text"] = problem["base_gt_full_text"]
            del problem["base_gt_full_text"]
            problem["base_response_text"] = problem["base_gt_response_text"]
            del problem["base_gt_response_text"]
            problem["base_reasoning_text"] = problem["base_gt_reasoning_text"]
            del problem["base_gt_reasoning_text"]
            problem["base_post_reasoning"] = problem["base_gt_post_reasoning"]
            del problem["base_gt_post_reasoning"]
        except KeyError:
            pass

    print(f"Loaded {len(data)} problems from {filename}")
    return data


if __name__ == "__main__":
    data = load_all_problems()
    # Structure of data[i]:
    #    'question_with_cue': str, question with hint
    #    'cue_ranges': list[list], e.g., [[3, 29]] in 'full_text' this describes the tokens where the hint is
    #    'question': str, question without hint
    #    'full_text': str, question with hint + reasoning + answer
    #    'reasoning_text': str, reasoning, between "<think>" and "</think>", excluding the <think> tags themselves
    #    'response_text': str, reasoning + answer
    #    'post_reasoning': str, answer
    #    'base_full_text': str, non-hinted question, non-hinted reasoning, non-hinted answer
    #    'base_response_text': str, non-hinted reasoning + non-hinted answer
    #    'base_reasoning_text': str, non-hinted reasoning
    #    'base_post_reasoning': str, non-hinted answer
    #    'gt_answer': str, the correct answer
    #    'cue_answer': str, the hinted answer
    #    'pn': int, the problem number

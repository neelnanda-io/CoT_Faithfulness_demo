from load_CoT import load_all_problems

from rollouts import RolloutsClient

if __name__ == "__main__":

    data = load_all_problems()
    problem_data = data[0]
    # Create client with default settings
    client = RolloutsClient(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", temperature=0.7, max_tokens=4096
    )

    rollouts = client.generate(
        problem_data["question_with_cue"],
        n_samples=10,
    )
    for response in rollouts:
        print("----------")
        print(f"{response.content[20:]=}")
        print(f"{response.reasoning[:20]=}")

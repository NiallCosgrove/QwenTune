import ollama
import json
import concurrent.futures
from tqdm import tqdm

# Configurations
EVALUATORS = ["qwen2.5", "mistral"]  # Local models for scoring
TOP_PERCENT_KEEP = 20  # Keep top 10% of responses
INPUT_FILE = "generated_chat_dataset.jsonl"
OUTPUT_FILE = "filtered_chat_dataset.jsonl"

# Load dataset
def load_dataset(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

# Function to get scores using Ollama
def get_scores(q1, r1, q2, r2, model):
    prompt = f"""
    <|system|> You are an AI evaluator. Score the following Q/A pairs:
    - Score 0 if response is factually incorrect.
    - Otherwise, score 1-10 for Response Quality (accuracy, coherence, fluency).
    - Score 1-10 for Conversational Flow (does second Q/A follow logically?).

    <|user|> {q1}
    <|assistant|> {r1}

    <|user|> {q2}
    <|assistant|> {r2}

    Scores:
    - Response Quality:
    - Conversational Flow:
    """

    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        text = response["message"]["content"]
        
        # Extract scores
        scores = [int(s) for s in text.split() if s.isdigit()]
        if len(scores) == 2:
            return scores[0], scores[1]  # Response Quality, Flow Score
    except Exception as e:
        print(f"Error with {model}: {e}")
        return None

# Evaluate dataset with sliding window
def evaluate_dataset(lines):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pair = {}

        for conv in lines:
            messages = conv["messages"]
            for i in range(0, len(messages) - 3, 2):  # Sliding window (Q1/R1, Q2/R2)
                q1, r1, q2, r2 = messages[i]["content"], messages[i+1]["content"], messages[i+2]["content"], messages[i+3]["content"]

                for model in EVALUATORS:
                    future = executor.submit(get_scores, q1, r1, q2, r2, model)
                    future_to_pair[future] = (q1, r1, q2, r2)

        for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(future_to_pair)):
            q1, r1, q2, r2 = future_to_pair[future]
            try:
                result = future.result()
                if result is not None:
                    response_score, flow_score = result
                    results.append((q1, r1, q2, r2, response_score, flow_score))
                    
                    # Debug Print: High-Scoring Pairs
                    if response_score >= 8 and flow_score >= 8:
                        print("\nHigh-Scoring Response")
                        print(f"Q1: {q1}\nA1: {r1}\nQ2: {q2}\nA2: {r2}")
                        print(f"Scores - Response: {response_score}, Flow: {flow_score}")

                    # Debug Print: Responses that got a Zero
                    if response_score == 0 or flow_score == 0:
                        print("\nZero-Scoring Response")
                        print(f"Q1: {q1}\nA1: {r1}\nQ2: {q2}\nA2: {r2}")
                        print(f"Scores - Response: {response_score}, Flow: {flow_score}")
                else:
                    print("Skipping invalid score from model")

            except Exception as e:
                print(f"Error processing a response: {e}")

    return results

# Filter top X% of responses
def filter_top_responses(results, top_percent):
    results.sort(key=lambda x: (x[4] + x[5]) / 2, reverse=True)  # Sort by avg score
    num_keep = int(len(results) * (top_percent / 100))
    return results[:num_keep]

# Save dataset in JSONL format
def save_dataset(results, filename):
    with open(filename, "w") as f:
        for q1, r1, q2, r2, response_score, flow_score in results:
            json.dump({
                "messages": [
                    {"role": "user", "content": q1},
                    {"role": "assistant", "content": r1},
                    {"role": "user", "content": q2},
                    {"role": "assistant", "content": r2}
                ],
                "response_score": response_score,
                "flow_score": flow_score
            }, f)
            f.write("\n")

# Main execution
if __name__ == "__main__":
    print("Loading dataset...")
    dataset_lines = load_dataset(INPUT_FILE)

    print("Evaluating dataset with Ollama scoring...")
    scored_results = evaluate_dataset(dataset_lines)

    print("Filtering top responses...")
    filtered_results = filter_top_responses(scored_results, TOP_PERCENT_KEEP)

    print(f"Saving final dataset ({len(filtered_results)} samples)...")
    save_dataset(filtered_results, OUTPUT_FILE)

    print("Dataset processing complete!")



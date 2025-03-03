import requests
import json

# Define the system prompt with guidelines.
system_prompt = (
    "You are an AI trained to generate high-quality, structured instruction-response datasets. "
    "Your task is to produce output in strict ChatML format. Each conversation must include three messages: "
    "a system message with these instructions, a user message with a prompt, and an assistant message with the answer. "
    "Responses must be concise and factually correct. Do not include any extra commentary or dialogue outside the ChatML format. "
    "Ensure that the output does not exceed a maximum sequence length of 16,000 tokens. "
    "Example ChatML format:\n"
    "system: [Instructions]\n"
    "user: [Question]\n"
    "assistant: [Answer]\n"
    "Output must be valid JSON."
)

# Function to read prompts from a text file.
def read_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        # Each non-empty line is treated as a separate prompt.
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

# URL for the local Ollama server API.
ollama_url = "http://localhost:11434/api/v1/generate"

def generate_conversation(system_msg, user_msg):
    """
    Sends a request to the local Ollama server with the system prompt and a user prompt.
    Returns the generated conversation in ChatML format.
    """
    payload = {
        "model": "qwen2.5",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    }
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()
        # Expect the API to return a JSON with a 'completion' key containing the ChatML conversation.
        if 'completion' in result:
            return result['completion']
        else:
            print("Unexpected response format:", result)
            return None
    except Exception as e:
        print("Error during API request:", e)
        return None

def main():
    # Read prompts from the text file.
    prompts = read_prompts("prompts.txt")
    all_conversations = []
    
    # Iterate over each prompt to generate a conversation.
    for prompt in prompts:
        print(f"Generating data for prompt: {prompt}")
        conversation = generate_conversation(system_prompt, prompt)
        if conversation:
            try:
                # Attempt to parse conversation as JSON.
                chatml = json.loads(conversation)
            except json.JSONDecodeError:
                # If not valid JSON, store it as a raw string.
                chatml = conversation
            all_conversations.append(chatml)
        else:
            print(f"Failed to generate data for prompt: {prompt}")
    
    # Consolidate all conversations into a single JSON object.
    dataset = {"conversations": all_conversations}
    
    # Save the dataset to a JSON file.
    with open("finetuning_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Dataset generated and saved to 'finetuning_dataset.json'.")

if __name__ == "__main__":
    main()

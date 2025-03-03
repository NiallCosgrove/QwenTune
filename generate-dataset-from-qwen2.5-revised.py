import requests
import json

def read_prompts(file_path):
    """Read non-empty prompts from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# URL for the local Ollama server API.
ollama_url = "http://localhost:11411/api/chat"

def generate_conversation(user_prompt):
    """
    Constructs a simple user prompt that instructs the model to generate
    a conversation in ChatML format and sends it to the local Ollama server.
    """
    full_prompt = (
        "Generate a conversation in ChatML format. "
        "The conversation must consist of exactly three messages: "
        "a system message, a user message, and an assistant message. "
        "The system message should be minimal (e.g., 'ChatML session started.'). "
        "The user message should be: '" + user_prompt + "' "
        "and the assistant message should provide a concise, factual answer. "
        "Output the entire conversation as valid JSON without any extra commentary."
    )
    
    payload = {
        "model": "qwen2.5",
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()
        # Expecting a JSON with a key "completion" containing the ChatML conversation.
        if "completion" in result:
            return result["completion"]
        else:
            print("Unexpected response format:", result)
            return None
    except Exception as e:
        print("Error during API request:", e)
        return None

def main():
    # Read prompts from the text file.
    prompts = read_prompts("new-prompts.txt")
    all_conversations = []
    
    for prompt in prompts:
        print(f"Processing prompt: {prompt}")
        conversation = generate_conversation(prompt)
        if conversation:
            try:
                # Attempt to parse the output as JSON.
                chatml = json.loads(conversation)
            except json.JSONDecodeError:
                chatml = conversation  # Use as raw string if JSON parsing fails.
            all_conversations.append(chatml)
        else:
            print(f"Failed to generate conversation for prompt: {prompt}")
    
    # Consolidate all conversations into a single JSON object.
    dataset = {"conversations": all_conversations}
    
    # Save the dataset to a JSON file.
    with open("finetuning_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Dataset saved to 'finetuning_dataset.json'.")

if __name__ == "__main__":
    main()

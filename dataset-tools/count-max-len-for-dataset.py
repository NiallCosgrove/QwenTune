from datasets import load_from_disk
from transformers import AutoTokenizer

# Load dataset from disk
dataset_path = "./datasets/alpaca"  # Adjust if needed
dataset = load_from_disk(dataset_path)["train"]  # Only split is 'train'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")  # Adjust model name

# Get max tokenized length
max_tokens = max(len(tokenizer(ex["text"])["input_ids"]) for ex in dataset)

print(f"Max tokenized length in dataset: {max_tokens}")


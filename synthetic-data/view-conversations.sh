jq -r '.messages[] | select(.role=="user" or .role=="assistant") | "\(.role): \(.content)\n"' generated_chat_dataset.jsonl | less

#!/bin/bash

while read prompt; do
    echo "Processing: $prompt"
    echo "$prompt" | ollama run qwen2.5 >> dataset.json
    echo "," >> dataset.json  # Add a comma between entries (cleanup needed after)
done < prompts.txt


echo "[" > final_dataset.json
sed '$ s/,$//' dataset.json >> final_dataset.json  # Remove trailing comma
echo "]" >> final_dataset.json

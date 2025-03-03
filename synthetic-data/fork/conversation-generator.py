import ollama
import json
import random
from tqdm import tqdm
import numpy.random as random

# Configurations
MODEL = "qwen2.5"
NUM_CONVERSATIONS = 10  # Total conversations to generate
MAX_TURNS = 10  # Prevent infinite loops
OUTPUT_FILE = "generated_chat_dataset.jsonl"

#temperature centered at 0.5 with 1% tails

# Generate a conversation starter
def generate_starter():
    prompt = """
    <|system|> Generate a direct, thought-provoking conversation starter. The conversation starter should mention the subject under discussion. The question should come from an intelligent academic in an unrelated field, seeking insight from an expert. 

    The question should reflect curiosity, assume intelligence, and invite an in-depth response from the expert. It can either require a simple factual answer or instead encourage explanation and analysis of a particular subject. 

    Output ONLY the question, nothing else. Thank you.
    """

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options = {"temperature":random.normal(loc=0.5,scale=0.1717)})
    #print(f"Response: {response}")
    return response["message"]["content"].strip()

# Generate follow-up questions
def generate_followups(conversation):
    last_question = conversation[-1]["content"]
    prompt = f"<|system|> Given this user question, list natural follow-up questions (if any). No preamble just the list. Do not number the list. Each follow-up should be on a new line.\nUser: {last_question}"
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options = {"temperature":random.normal(loc=0.5,scale=0.1717)})
    followups = response["message"]["content"].strip().split("\n")  # Assumes line-separated responses
    #print(f"FOLLOWUPS: {followups}")
    return [q.strip() for q in followups if q.strip()]

# Generate assistant responses
def generate_response(question):
    prompt = f"""
    <|system|> You are a world-renowned expert in the topic under discussion. You are speaking to an academic in an unrelated field. Provide a confident, detailed, and structured response, assuming intelligence in your listener but limited prior subject knowledge. 

    Respond naturally as a human expert would, no need to be terse, give the subject matter full treatment with an eye for nuance — do not hedge, do not state that you are an AI, and do not disclaim knowledge. Instead, provide a thoughtful, well-reasoned, and articulate answer, rich in detail and insight.

    <|user|> {question}
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options = {"temperature":random.normal(loc=0.5,scale=0.1717)})
    return response["message"]["content"].strip()

# Generate full conversation with variable length
def generate_conversation():
    conversation = [{"role": "user", "content": generate_starter()}]

    for _ in range(MAX_TURNS):
        assistant_reply = generate_response(conversation[-1]["content"])
        conversation.append({"role": "assistant", "content": assistant_reply})

        followups = generate_followups(conversation)
        if not followups:
            break  # Stop if no follow-ups

        conversation.append({"role": "user", "content": random.choice(followups)})

        # Ensure every user message has a paired assistant response
        assistant_reply = generate_response(conversation[-1]["content"])
        conversation.append({"role": "assistant", "content": assistant_reply})

    return conversation


# Generate dataset
dataset = []
for _ in tqdm(range(NUM_CONVERSATIONS), desc="Generating conversations"):
    dataset.append({"messages": generate_conversation()})

# Save dataset
with open(OUTPUT_FILE, "w") as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write("\n")

print(f"✅ Generated {NUM_CONVERSATIONS} conversations with variable lengths!")

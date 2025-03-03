import os
import json
import numpy as np
from datetime import datetime

import traceback
import argparse
import logging


import torch

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_logging():
    logging.basicConfig(
        filename="lm_eval.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def load_model(model_name: str):
    """Load the model once and ensure proper tokenizer settings."""
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #print("DEBUG: Model class before wrapping:", model.__class__.__name__)
    model = model.to("cuda")
    
    # Ensure proper tokenizer settings
    if tokenizer.pad_token is None or tokenizer.pad_token == "<|vision_pad|>":
        logging.info("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    if tokenizer.eos_token_id is None:
        logging.info("Setting eos_token_id.")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    
    if tokenizer.bos_token_id is None:
        logging.info("Setting bos_token_id.")
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    
    return model, tokenizer

def frange(start, stop, step):
    """Floating point range generator."""
    while start <= stop:
        yield round(start, 2)
        start += step

def sanitize_json(data):
    """Recursively sanitizes a dictionary or list, ensuring all values are JSON-serializable."""
    if isinstance(data, dict):
        return {key: sanitize_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(value) for value in data]
    elif isinstance(data, torch.dtype):  # Convert torch dtype to string
        return str(data)
    elif isinstance(data, np.generic):  # Convert NumPy scalars to Python types
        return data.item()
    elif isinstance(data, np.ndarray):  # Convert NumPy arrays to lists
        return data.tolist()
    else:
        return data  # Return as-is if it's already serializable


def evaluate_with_model(model, tokenizer, model_name, task, temp_range, num_tests, num_fewshot):
    """Evaluate using a preloaded model and save partial results."""
    
    results = {}  
    results_file = f"lm_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for temp in temp_range:
        logging.info(f"✅ Running evaluation with task={task}, temperature={temp}, fewshot={num_fewshot}")

        #simple_evaluate expects a csv string of kwargs
        my_kwargs = f"temperature={temp},do_sample=True"
        
        #wrap the model to keep lm-eval happy
        wrapped_model = HFLM(pretrained=model_name, batch_size=1, device="cuda")

        eval_result = simple_evaluate(
            model=wrapped_model,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=num_tests,
            gen_kwargs=my_kwargs
        )

        results[temp] = eval_result

        # Example usage before dumping:
        clean_results = sanitize_json(results)

        # Save results after each temp iteration
        with open(results_file, "w") as f:
            json.dump(clean_results, f, indent=4, default=sanitize_json)
        
        logging.info(f"Partial results saved to {results_file}")
        print(f"✅ Saved progress: {results_file}")

    logging.info(f"✅ Final evaluation complete! Results saved to {results_file}")
    print(f"✅ Final results saved: {results_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model path (local or Hugging Face)")
    parser.add_argument("--task", type=str, default="gsm8k", help="Evaluation task")
    parser.add_argument("--temp-start", type=float, default=0.3, help="Starting temperature")
    parser.add_argument("--temp-end", type=float, default=0.8, help="Ending temperature")
    parser.add_argument("--temp-step", type=float, default=0.1, help="Temperature step size")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of test samples per task")
    parser.add_argument("--num-fewshot", type=int, default=5, help="Number of few-shot examples per task")
    
    args = parser.parse_args()
    temp_range = list(frange(args.temp_start, args.temp_end, args.temp_step))
    
    model, tokenizer = load_model(args.model_name)
    try:
        evaluate_with_model(model, tokenizer, args.model_name, args.task, temp_range, args.num_tests, args.num_fewshot)
    except AttributeError as e:
        print("DEBUG: AttributeError caught!")
        print(traceback.format_exc())  # ✅ Print the full traceback, not just the last line

if __name__ == "__main__":
    main()





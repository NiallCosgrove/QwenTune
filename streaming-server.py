import torch
import threading
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
import random
import signal
import sys
import asyncio

app = FastAPI()

# -------------------------------------------------------------------------
# Argument Parsing for Device Selection
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Force GPU usage (fails if unavailable).")
parser.add_argument("--cpu", action="store_true", help="Force CPU usage.")
parser.add_argument("--model", help="Model to load. i.e. --model ./model", default="./model-r1")
args = parser.parse_args()

# Prevent conflicting arguments
if args.gpu and args.cpu:
    print("Error: Cannot specify both --gpu and --cpu.")
    exit(1)

# Auto-detect device if no flag is provided
if args.gpu:
    if not torch.cuda.is_available():
        print("Error: --gpu specified, but CUDA is not available.")
        exit(1)
    device = torch.device("cuda")
    print("Using GPU")
elif args.cpu or not torch.cuda.is_available():
    device = torch.device("cpu")
    print("Using CPU")
else:
    device = torch.device("cuda")
    print("Auto-selecting GPU")

# -------------------------------------------------------------------------
# Load Model and Tokenizer (Vanilla HF for CPU, Unsloth for GPU)
# -------------------------------------------------------------------------
model_name = args.model
print(f"Loading model: {model_name} on {device}")

if device.type == "cuda":
    from unsloth import FastLanguageModel

    def load_model_gpu():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=16384,
            dtype="float16",
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    
    model, tokenizer = load_model_gpu()
else:
    def load_model_cpu():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        return model, tokenizer
    
    model, tokenizer = load_model_cpu()

# Move model to selected device
if device.type == "cuda":
    model.to(device)

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.pop("prompt", "Hello world")  # Remove 'prompt' before passing kwargs
    seed = body.get("seed", random.randint(0, 2**32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    text_streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        decode_kwargs={"skip_special_tokens": False}
    )
    
    generation_kwargs = body  # 'prompt' has already been removed
    generation_kwargs.update({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "streamer": text_streamer,
        "max_new_tokens": body.get("max_new_tokens", 512),  # Increase default
        "early_stopping": body.get("early_stopping", False),
        "forced_eos_token_id": tokenizer.eos_token_id,
    })
    
    # Directly call model.generate with full kwargs
    thread = threading.Thread(target=lambda: model.generate(**generation_kwargs))
    thread.start()
    
    async def event_stream():
        for token_text in text_streamer:
            yield token_text
        print("🟢 Generation complete.")  # Notify when done
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.post("/kill")
async def shutdown():
    """Gracefully shuts down the server from a remote request."""
    print("Received shutdown request. Stopping server...")
    task = asyncio.create_task(stop_server())
    await task
    return {"status": "shutting down"}

async def stop_server():
    """Shutdown Uvicorn cleanly."""
    loop = asyncio.get_event_loop()
    loop.stop()

# Handle CTRL-C shutdown
def handle_sigterm(signal, frame):
    print("Received SIGTERM. Stopping server...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigterm)  # CTRL-C
signal.signal(signal.SIGTERM, handle_sigterm)  # Docker/Kubernetes shutdown signal

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# -------------------------------------------------------------------------
# List of supported generation kwargs for future tweaking (Full Set)
# -------------------------------------------------------------------------
# max_new_tokens          # Maximum number of new tokens to generate
# temperature            # Sampling temperature for randomness
# top_k                  # Restrict sampling to top-k likely tokens
# top_p                  # Restrict sampling to top-p cumulative probability
# repetition_penalty     # Penalize repeated tokens
# do_sample              # Enable/disable sampling
# num_return_sequences   # Number of sequences to generate
# length_penalty         # Adjust length preference for beam search
# early_stopping         # Stop beam search early if all beams reach EOS
# no_repeat_ngram_size   # Prevent repeating n-grams of given size
# min_length             # Minimum length of generated sequence
# seed                   # Random seed for reproducibility
# logit_bias             # Bias logit probabilities of specific tokens
# typical_p              # Typical decoding probability threshold
# decoder_start_token_id # Token ID to start decoding
# forced_bos_token_id    # Force a specific beginning-of-sequence token
# forced_eos_token_id    # Force a specific end-of-sequence token



######!/usr/bin/env python3   use a venv aware version of this!!!  we don't just need python3 we need exactly 3.11
import argparse
import os
import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer
from unsloth import standardize_sharegpt


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model with Unsloth support.")
    # Model & dataset paths
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the base model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory of the processed dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the fine-tuned model")
    # Training parameters
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8, help="Training batch size per device (default: %(default)s)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps (default: %(default)s)")
    # Subset training control
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="Fraction of dataset to use (default: %(default)s)")
    # Tokenization & training behaviour
    parser.add_argument("--max-length", type=int, default=2048, help="Max token length (default: %(default)s)")
    parser.add_argument("--save-steps", type=int, default=1000, help="Checkpoint save interval (default: %(default)s)")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging interval (default: %(default)s)")
    # Resume training
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume from latest checkpoint or specify checkpoint folder")
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recent checkpoint in the directory."""
    checkpoints = [
        os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        return None  # No checkpoints found
    return max(checkpoints, key=lambda x: int(x.split("checkpoint-")[-1]))

def print_training_summary(args):
    """Prints a boxed summary of the training configuration."""
    summary = [
        ("Model", args.model_dir),
        ("Dataset", args.dataset_dir),
        ("Subset Ratio", f"{args.subset_ratio} (Full Dataset)" if args.subset_ratio == 1.0 else f"{args.subset_ratio}"),
        ("Max Token Length", args.max_length),
        ("Batch Size", args.per_device_train_batch_size),
        ("Gradient Steps", args.gradient_accumulation_steps),
        ("Epochs", args.num_train_epochs),
        ("Save Steps", args.save_steps),
        ("Logging Steps", args.logging_steps),
        ("Using Unsloth", "Yes"),
    ]
    # Determine box width
    box_width = max(len(f"{label}: {value}") for label, value in summary) + 40

    print("+" + "-" * (box_width - 1) + "+")
    print(f"|{'TRAINING SUMMARY'.center(box_width - 1)}|")
    print("+" + "-" * (box_width - 1) + "+")
    
    for label, value in summary:
        print(f"| {label.ljust(20)} {str(value).ljust(box_width - 24)} |")

    print("+" + "-" * (box_width - 1) + "+")

def preprocess_function(example, tokenizer, max_length=2048):
    # Tokenize text with truncation
    text = example.get("text", "")
    return tokenizer(text, truncation=True, max_length=max_length)  #, padding="max_length")

def load_unsloth_model(args):
    # Ensure CUDA is available for Unsloth
    if not torch.cuda.is_available():
        raise RuntimeError("Unsloth requires CUDA, but no GPU was detected.")
    
    from unsloth import FastLanguageModel
    import unsloth.models._utils
    # Workaround because SFTTrainer() needs logits and unsloth wants to unsloth
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # Patch _unsloth_get_batch_samples to handle batch sample extraction correctly
    import inspect
    def patched_get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None

        # NIALLDEBUG: Check if model allows **kwargs
        model = self.model
        # The following line attempts to fetch the forward method; alternative method commented out:
        # f = model.base_model.model.forward if hasattr(model, "base_model") else model.forward
        f = getattr(model.base_model, "model", model).forward
        has_kwargs = tuple(inspect.signature(f).parameters.values())[-1].kind == inspect._VAR_KEYWORD

        # Iterate to collect batches
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        # Calculate number of items in batch if possible
        if has_kwargs and len(batch_samples) > 0 and "labels" in batch_samples[0]:
            try:
                num_items_in_batch = sum(
                    [(x["labels"][..., 1:] != -100).sum() for x in batch_samples]
                )
                # If averaging tokens across devices is enabled, gather counts from all devices
                if self.args.average_tokens_across_devices:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.item()
            except Exception as exception:
                # NIALLDEBUG: Exception encountered during batch sample processing
                pass

        return batch_samples, num_items_in_batch

    # Apply the patch
    unsloth.models._utils._unsloth_get_batch_samples = patched_get_batch_samples

    print("Using Unsloth for faster fine-tuning")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_length,     # we need a long context dataset for qwens 32768 - but we don't have one, 1119 is max seq in our dataset.
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False
    )
        
    # Do not compile Unsloth models since they are quantized
    return model, tokenizer

def load_model(args):
    # Always load Unsloth model in this version
    return load_unsloth_model(args)


def rename_messages_to_conversations(example):
    example["conversations"] = example.pop("messages")  # Rename key
    return example


def convert_to_qwen_chatml(example):
    """
    Converts dataset examples into Qwen's expected ChatML format.
    """
    chatml_conversations = []
    
    # Ensure system message is present (Qwen auto-adds it if missing)
    if example["conversations"][0]["role"] != "system":
        chatml_conversations.append(
            {"role": "system", "content": "You are a helpful assistant."}
        )
    
    # Convert conversations into ChatML format
    for msg in example["conversations"]:
        chatml_conversations.append({"role": msg["role"], "content": msg["content"]})
    
    return {"conversations": chatml_conversations}


def load_or_tokenize_dataset(dataset_dir, tokenizer, subset_ratio=1.0, max_length=2048):
    """Loads a pre-tokenized dataset if available, otherwise tokenizes, saves it, and applies subset selection."""
    
    tokenized_path = os.path.join(dataset_dir, "tokenized")

    # Check if a pre-tokenized dataset exists
    if os.path.exists(tokenized_path):
        print(f"🔹 Loading pre-tokenized dataset from {tokenized_path}")
        dataset = load_from_disk(tokenized_path)
    else:
        print("⚡ Tokenizing dataset (this may take a while)...")
        
        #Load raw dataset & apply transformations
        dataset = load_from_disk(dataset_dir)["train"]
        dataset = dataset.map(rename_messages_to_conversations)
        dataset = standardize_sharegpt(dataset)
        dataset = dataset.map(convert_to_qwen_chatml)


    if subset_ratio < 1.0:
        subset_size = int(subset_ratio * len(dataset))
        dataset = dataset.select(range(subset_size))
        
    
    #4 Tokenize and save
    dataset = dataset.map(lambda ex: preprocess_function(ex, tokenizer, max_length), batched=True)
    dataset.save_to_disk(tokenized_path)
    print(f"✅ Tokenized dataset saved to {tokenized_path}")

    return dataset


def main():
    args = parse_args()

    # Handle checkpoint selection
    checkpoint_path = None
    checkpoint_dir = args.output_dir
    if args.resume:
        if args.resume == "latest":
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise ValueError("No checkpoints found, cannot resume training.")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, args.resume)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint {checkpoint_path} does not exist.")

    # Load model and tokenizer using Unsloth
    model, tokenizer = load_model(args)
    train_dataset = load_or_tokenize_dataset(args.dataset_dir, tokenizer, args.subset_ratio, args.max_length)
    print(f"Training dataset size: {len(train_dataset)}")
    
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        #fp16=False if torch.cuda.is_available() else True,    #legacy from PEFT cpu support
        bf16=True if torch.cuda.is_available() else False,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=True,     #oxford comma ^^
    )

    ###############################
    #🎵little boxes little boxes🎵#
    ###############################
    print_training_summary(args)
    
    model.train()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=bool(args.resume))  #pythonic to the max ^^
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()

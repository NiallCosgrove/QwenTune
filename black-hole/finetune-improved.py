    










import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model with optional Unsloth support.")

    # Model & dataset paths
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the base model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory of the processed dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the fine-tuned model")

    # Training params
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8, help="Training batch size per device (default: %(default)s)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps (default: %(default)s)")

    # Subset training control
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="Fraction of dataset to use (default: %(default)s)")

    # Tokenization & training behavior
    parser.add_argument("--max-length", type=int, default=1120, help="Max token length (default: %(default)s)")
    parser.add_argument("--save-steps", type=int, default=4000, help="Checkpoint save interval (default: %(default)s)")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval (default: %(default)s)")

    # Enable Unsloth
    parser.add_argument("--use-unsloth", action="store_true", help="Enable Unsloth for faster training - CUDA required.")

    #resume
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
        ("Using Unsloth", "Yes" if args.use_unsloth else "No"),
    ]
    
    # Determine box width
    box_width = max(len(f"{label}: {value}") for label, value in summary) + 40

    # Print boxed summary
    print("+" + "-" * (box_width - 1) + "+")
    print(f"|{'TRAINING SUMMARY'.center(box_width - 1)}|")
    print("+" + "-" * (box_width - 1) + "+")
    
    for label, value in summary:
        print(f"| {label.ljust(20)} {str(value).ljust(box_width - 24)} |")

    print("+" + "-" * (box_width - 1) + "+")

def preprocess_function(example, tokenizer, max_length=1120):  
    text = example.get("text", "")
    return tokenizer(text, truncation=True, max_length=max_length) #, padding="max_length")

def load_model(args):
    if args.use_unsloth:
        return load_unsloth_model(args)  # ✅ Unsloth loads the model internally
    else:
        model, tokenizer = load_peft_model(args)  # ✅ Now always wrapped correctly
        #model = torch.compile(model)  #this fails with ValueError: No columns in the dataset match the model's forward method signature. The following columns have been ignored: [output, input, input_ids, instruction, messages, attention_mask, text]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.
        return model, tokenizer

def load_peft_model(args):
    print("Using PEFT for fine-tuning")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.config.use_cache = False
    lora_config = get_lora_config(model)    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: prints number of trainable parameters.
    return model, tokenizer


def load_unsloth_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Unsloth requires CUDA, but no GPU was detected.")
    
    from unsloth import FastLanguageModel
    import unsloth.models._utils

    # Patch `_unsloth_get_batch_samples`
    import inspect
    def patched_get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None

        # Check if model allows **kwargs
        model = self.model
        #NIALLDEBUG
        f = getattr(model.base_model, "model", model).forward
        #f = model.base_model.model.forward if hasattr(model, "base_model") else model.forward
        has_kwargs = tuple(inspect.signature(f).parameters.values())[-1].kind == inspect._VAR_KEYWORD

        # Iterate to find all batches
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break
        pass
        # Get num_items_in_batch
        if has_kwargs and len(batch_samples) > 0 and "labels" in batch_samples[0]:
            try:
                num_items_in_batch = sum(
                    [(x["labels"][..., 1:] != -100).sum() for x in batch_samples]
                )

                if self.args.average_tokens_across_devices:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.item()

            except Exception as exception:
                logger.warning_once(exception)
        pass

        return batch_samples, num_items_in_batch
    pass


    # Apply the patch
    unsloth.models._utils._unsloth_get_batch_samples = patched_get_batch_samples

    print("Using Unsloth for faster fine-tuning")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=1120,     #we need a long context dataset for qwens 32768 - but we don't have one.
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False
    )
        
    # Do not compile Unsloth models since they are quantized  # Enable torch.compile() for speed
    return model, tokenizer

import os
from datasets import load_from_disk, load_dataset

def load_or_tokenize_dataset(dataset_dir, tokenizer, subset_ratio=1.0, max_length=1120):
    """Loads a pre-tokenized dataset if available, otherwise tokenizes and saves it."""

    tokenized_path = os.path.join(dataset_dir, "tokenized")

    # Check if pre-tokenized dataset exists
    if os.path.exists(tokenized_path):
        print(f"🔹 Loading pre-tokenized dataset from {tokenized_path}")
        dataset = load_from_disk(tokenized_path)
    else:
        print(f"⚡ Tokenizing dataset (this may take a while)...")

        # Load dataset from Arrow or JSON
        if os.path.isdir(dataset_dir) and "train" in os.listdir(dataset_dir):  
            dataset = load_from_disk(dataset_dir)["train"]  # Load Arrow format
        else:
            dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "data.json"))["train"]  # Load JSON

        # Apply tokenization
        dataset = dataset.map(lambda ex: preprocess_function(ex, tokenizer, max_length), batched=True)

        # Save tokenized dataset
        dataset.save_to_disk(tokenized_path)
        print(f"✅ Tokenized dataset saved to {tokenized_path}")

    # Handle subset selection
    if subset_ratio < 1.0:
        subset_size = int(subset_ratio * len(dataset))
        dataset = dataset.select(range(subset_size))
        print(f"🔹 Using subset: {subset_size} samples")

    return dataset

def get_lora_config(model):
    """Automatically detect LoRA-compatible layers."""
    #target_modules = [name for name, _ in model.named_modules() if "proj" in name]
    target_modules = ["q_proj", "v_proj"]  # Restore hardcoded values for testing
    
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,  # Dynamically detected
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )



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

    # Load model and tokenizer  
    model, tokenizer = load_model(args)
    model.tie_weights()
    train_dataset = load_or_tokenize_dataset(args.dataset_dir, tokenizer, args.subset_ratio, args.max_length)

    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=False if torch.cuda.is_available() else True,   #PEFT doesn't like bf16
        bf16=True if torch.cuda.is_available() else False, 
        logging_steps=args.logging_steps,  # Now configurable
        save_steps=args.save_steps,  # Now configurable
        save_total_limit=4,
        evaluation_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=True,    
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    ###############################
    #🎵little boxes little boxes🎵#
    print_training_summary(args)  #
    ###############################


    # Start training
    model.train()
    trainer.train(resume_from_checkpoint=checkpoint_path)
    #trainer.train()

    # Save the fine-tuned model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()

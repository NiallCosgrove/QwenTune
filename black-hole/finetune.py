import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorWithPadding
from transformers import DataCollatorForLanguageModeling

def preprocess_function(example, tokenizer, max_length=1024):
    text = example.get("text", "")
    return tokenizer(text, truncation=True, max_length=max_length)

def main():
    parser = argparse.ArgumentParser(description="Fine tune model using LoRA on our dataset")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the base model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    args = parser.parse_args()

    # Set up device and load model and tokenizer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model = model.to(device)

    # Configure LoRA for parameter-efficient fine tuning.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust based on your model's architecture.
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: prints number of trainable parameters.

    # Load and preprocess the dataset.
    # for testing we will only train on % of the full dataset
    full_dataset = load_from_disk(args.dataset_dir)
    full_train_dataset = full_dataset["train"] if "train" in full_dataset else full_dataset

    subset_size = int(0.01 * len(full_train_dataset))  # 1% of the dataset
    subset_dataset = full_train_dataset.shuffle(seed=42).select(range(subset_size))

    train_dataset = subset_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)

    # Create data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none"
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Start training.
    trainer.train()

    # Save the fine tuned model.
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

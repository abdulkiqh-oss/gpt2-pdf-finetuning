import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
parser.add_argument("-m", "--model-name", default="gpt2", help="Base model name or path (default: gpt2)")
parser.add_argument("-d", "--data-file", default="data.txt", help="Input training dataset file (default: data.txt)")
parser.add_argument("-o", "--output-dir", default="./gpt2-full-model", help="Directory for checkpoint outputs (default: ./gpt2-full-model)")
parser.add_argument("-s", "--save-dir", default="./my_full_gpt2", help="Final model output directory (default: ./my_full_gpt2)")
parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
parser.add_argument("-b", "--batch-size", type=int, default=2, help="Per-device train batch size (default: 2)")
parser.add_argument("-g", "--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
parser.add_argument("-lr", "--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
parser.add_argument("-w", "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
parser.add_argument("-l", "--max-length", type=int, default=256, help="Max sequence length for tokenization (default: 256)")
parser.add_argument("--use-cpu", action="store_true", help="Force CPU usage for training")
parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume training from checkpoint")

args = parser.parse_args()

# Initialize Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name)

for param in model.parameters():
    param.requires_grad = True

# Load and Tokenize Dataset
dataset = load_dataset("text", data_files={"train": args.data_file})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=args.max_length,
        padding="max_length"
    )

print("Preparing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    logging_steps=5,
    report_to="tensorboard",
    save_strategy="epoch",
    use_cpu=args.use_cpu
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train(resume_from_checkpoint=args.resume)

# Save Final Model and Tokenizer
model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"Full training completed successfully! Model saved to {args.save_dir}")
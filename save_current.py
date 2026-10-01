import argparse
import glob
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Save model weights from the latest training checkpoint.")
parser.add_argument("-c", "--checkpoints-dir", default="./gpt2-full-model", help="Directory containing checkpoints (default: ./gpt2-full-model)")
parser.add_argument("-o", "--output-dir", default="./my_full_gpt2", help="Directory to save final model (default: ./my_full_gpt2)")

args = parser.parse_args()

checkpoints_pattern = os.path.join(args.checkpoints_dir, "checkpoint-*")
checkpoints = glob.glob(checkpoints_pattern)

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading weights from: {latest_checkpoint}")
    
    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Successfully saved to {args.output_dir}!")
else:
    print(f"No checkpoints found in {args.checkpoints_dir}.")
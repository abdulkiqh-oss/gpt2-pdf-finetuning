import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description="Run inference using GPT-2 or custom model.")
parser.add_argument("-m", "--model", default="gpt2", help="Model name or path (default: gpt2)")
parser.add_argument("-p", "--prompt", default="Artificial Intelligence on Arch Linux is", help="Prompt text")
parser.add_argument("-l", "--max-length", type=int, default=50, help="Maximum length of generated text (default: 50)")
parser.add_argument("-n", "--num-sequences", type=int, default=1, help="Number of generated sequences (default: 1)")

args = parser.parse_args()

generator = pipeline('text-generation', model=args.model)

results = generator(args.prompt, max_length=args.max_length, num_return_sequences=args.num_sequences)

print(results[0]['generated_text'])
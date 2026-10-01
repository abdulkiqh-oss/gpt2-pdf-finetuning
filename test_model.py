import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description="Test a fine-tuned text generation model.")
parser.add_argument("-m", "--model", default="./my_full_gpt2", help="Path or name of the model (default: ./my_full_gpt2)")
parser.add_argument("-p", "--prompt", default="who i can use aircrack-ng", help="Input text prompt for generation")
parser.add_argument("-l", "--max-length", type=int, default=100, help="Maximum length of generated text (default: 100)")
parser.add_argument("-n", "--num-sequences", type=int, default=1, help="Number of return sequences (default: 1)")

args = parser.parse_args()

generator = pipeline('text-generation', model=args.model)

results = generator(args.prompt, max_length=args.max_length, num_return_sequences=args.num_sequences)

print("\n--- Output ---")
print(results[0]['generated_text'])
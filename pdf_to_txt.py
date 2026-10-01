from pypdf import PdfReader
import re
import argparse

# Setup command-line argument parser (--input and --output)
parser = argparse.ArgumentParser(description="Extract text from PDF files to a text file.")
parser.add_argument("-i", "--input", default="document.pdf", help="Input PDF file path (default: document.pdf)")
parser.add_argument("-o", "--output", default="data.txt", help="Output text file path (default: data.txt)")

args = parser.parse_args()

pdf_filename = args.input
output_filename = args.output

reader = PdfReader(pdf_filename)
extracted_text = []

print(f"Reading {len(reader.pages)} pages from: {pdf_filename}...")

for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        # Remove extra whitespace and duplicate blank lines
        clean_text = re.sub(r'\n+', '\n', text).strip()
        extracted_text.append(clean_text)

# Save cleaned text to output file
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n\n".join(extracted_text))

print(f"Text extracted successfully and saved to {output_filename}")
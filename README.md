# GPT-2 Custom Fine-Tuning & PDF Data Pipeline

A complete, modular Python pipeline designed to extract and clean unstructured text from PDF documents and fine-tune a Causal Language Model (GPT-2) using Hugging Face Transformers and PyTorch.

---

## Project Structure

├── pdf_to_txt.py      # Extract and clean text from PDF files
├── full_train.py      # Train/Fine-tune GPT-2 on processed text data
├── save_current.py    # Export the latest checkpoint to a final model directory
├── test_model.py      # Run inference on the fine-tuned model
├── run_gpt2.py        # Run inference on the base GPT-2 model
├── requirements.txt   # Python dependencies list
└── .gitignore         # Ignores virtual environments and large binary files

---

## Prerequisites

- Python 3.8+
- Git
- Git LFS (Optional: required only if tracking large .gguf or .safetensors model weights)

---

## Installation & Environment Setup

### 1. Freeze Dependencies
Before pushing to Git, activate your virtual environment and save your installed packages to requirements.txt:

pip freeze > requirements.txt

### 2. Clone Repository
git clone <YOUR_REPOSITORY_URL>
cd gpt2-project

### 3. Create & Activate Virtual Environment
- Linux / macOS:
  python3 -m venv venv
  source venv/bin/activate

- Windows (CMD):
  python -m venv venv
  venv\Scripts\activate

### 4. Install Requirements
pip install -r requirements.txt

---

## Complete Execution Guide

All Python scripts accept command-line arguments via argparse. You can run them with default values or pass custom flags.

### Step 1: Extract Text from PDF (pdf_to_txt.py)
Extract text content from a target PDF file and format it into a .txt file for dataset preparation.

# Default usage
python pdf_to_txt.py

# Custom input and output files
python pdf_to_txt.py --input document.pdf --output data.txt

# Short flags
python pdf_to_txt.py -i document.pdf -o data.txt

### Step 2: Fine-Tune GPT-2 (full_train.py)
Train/fine-tune the GPT-2 model on your extracted dataset.

# Default training
python full_train.py

# Advanced custom training parameters
python full_train.py \
  --data-file data.txt \
  --epochs 5 \
  --batch-size 2 \
  --grad-accum 4 \
  --learning-rate 5e-5 \
  --save-dir ./my_full_gpt2

### Step 3: Export Latest Checkpoint (save_current.py)
Extract and save the latest training checkpoint weights to a final standalone model directory.

# Default paths
python save_current.py

# Custom paths
python save_current.py --checkpoints-dir ./gpt2-full-model --output-dir ./my_full_gpt2

### Step 4: Test Fine-Tuned Model (test_model.py)
Run generation tests on your custom fine-tuned model.

# Default inference
python test_model.py

# Custom prompt and settings
python test_model.py --model ./my_full_gpt2 --prompt "How to use aircrack-ng" --max-length 100

### Step 5: Test Base Model (run_gpt2.py)
Run text generation directly on the base GPT-2 model.

# Default usage
python run_gpt2.py

# Custom prompt
python run_gpt2.py --prompt "Artificial Intelligence on Arch Linux is" --max-length 50

---

## Large File Management (.gitignore & Git LFS)

Large binary files and temporary model weights are excluded from Git tracking via .gitignore to maintain a clean repository structure.

If you need to track large model weights (e.g. .gguf files) using Git LFS:

git lfs install
git lfs track "*.gguf"
git add .gitattributes
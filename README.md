# 🚀 GPT-2 Custom Fine-Tuning & PDF Data Pipeline

A complete, modular Python pipeline for extracting and cleaning unstructured text from PDF documents and fine-tuning a **GPT-2 Causal Language Model** using **Hugging Face Transformers** and **PyTorch**.

The project is designed to keep the entire workflow simple, reproducible, and easy to extend.

---

## 📁 Project Structure

```text
gpt2-project/
│
├── 📄 pdf_to_txt.py        # Extract and clean text from PDF files
├── 🧠 full_train.py        # Fine-tune GPT-2 on processed text data
├── 💾 save_current.py      # Export the latest checkpoint
├── 🧪 test_model.py        # Test the fine-tuned model
├── 🤖 run_gpt2.py          # Run inference with the base GPT-2 model
├── 📦 requirements.txt     # Python dependencies
├── 🚫 .gitignore           # Ignore virtual environments and temporary files
└── 📘 README.md            # Project documentation
```

---

## 🔄 Pipeline Overview

```text
PDF Document
     │
     ▼
📄 pdf_to_txt.py
     │
     ▼
Clean Text Dataset
     │
     ▼
🧠 full_train.py
     │
     ▼
Training Checkpoints
     │
     ▼
💾 save_current.py
     │
     ▼
Final Fine-Tuned Model
     │
     ▼
🧪 test_model.py
     │
     ▼
Generated Text
```

---

## ✨ Features

* 📄 Extract text from PDF documents
* 🧹 Clean and normalize unstructured text
* 🧠 Fine-tune GPT-2 with PyTorch
* 🤗 Use Hugging Face Transformers
* 💾 Save and export training checkpoints
* 🧪 Test the fine-tuned model with custom prompts
* 🤖 Compare results with the original GPT-2 model
* ⚙️ Configure training and generation parameters through CLI arguments
* 📦 Keep the project modular and easy to maintain

---

## 🛠️ Prerequisites

Make sure the following are installed:

* **Python 3.8+**
* **Git**

A GPU with CUDA support is recommended for faster training, although CPU training may also be possible depending on the model size and dataset.

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd gpt2-project
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Freeze Dependencies

After installing and testing the project, update `requirements.txt` with the exact packages in your environment:

```bash
pip freeze > requirements.txt
```

> 💡 It is generally better to generate `requirements.txt` after confirming that the project works correctly in the virtual environment.

---

# 🚀 Complete Execution Guide

All scripts use `argparse`, so you can run them with default values or provide custom command-line arguments.

---

## 1️⃣ Extract Text from PDF

### `pdf_to_txt.py`

Extract and clean text from a PDF document and save it as a `.txt` dataset.

### Default Usage

```bash
python pdf_to_txt.py
```

### Custom Input and Output

```bash
python pdf_to_txt.py --input document.pdf --output data.txt
```

### Short Flags

```bash
python pdf_to_txt.py -i document.pdf -o data.txt
```

### Example

```text
Input:
document.pdf

Output:
data.txt
```

The generated text file is then used as the training dataset.

---

## 2️⃣ Fine-Tune GPT-2

### `full_train.py`

Train or fine-tune GPT-2 using the extracted text dataset.

### Default Training

```bash
python full_train.py
```

### Custom Training Parameters

```bash
python full_train.py \
    --data-file data.txt \
    --epochs 5 \
    --batch-size 2 \
    --grad-accum 4 \
    --learning-rate 5e-5 \
    --save-dir ./my_full_gpt2
```

### Main Parameters

| Argument          | Description                            |
| ----------------- | -------------------------------------- |
| `--data-file`     | Path to the training text file         |
| `--epochs`        | Number of training epochs              |
| `--batch-size`    | Training batch size                    |
| `--grad-accum`    | Gradient accumulation steps            |
| `--learning-rate` | Learning rate                          |
| `--save-dir`      | Directory where checkpoints are stored |

> ⚠️ Training parameters may need to be adjusted depending on dataset size, available VRAM, and model configuration.

---

## 3️⃣ Export the Latest Checkpoint

### `save_current.py`

Find the latest training checkpoint and export it into a standalone final model directory.

### Default Usage

```bash
python save_current.py
```

### Custom Paths

```bash
python save_current.py \
    --checkpoints-dir ./gpt2-full-model \
    --output-dir ./my_full_gpt2
```

### Example Directory Layout

```text
gpt2-full-model/
├── checkpoint-500/
├── checkpoint-1000/
└── checkpoint-1500/

            ↓

my_full_gpt2/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

> 📌 The exact generated files depend on the saving logic implemented in `save_current.py`.

---

## 4️⃣ Test the Fine-Tuned Model

### `test_model.py`

Run text-generation tests against your custom fine-tuned model.

### Default Usage

```bash
python test_model.py
```

### Custom Model and Prompt

```bash
python test_model.py \
    --model ./my_full_gpt2 \
    --prompt "How to use aircrack-ng" \
    --max-length 100
```

### Custom Example

```bash
python test_model.py \
    --model ./my_full_gpt2 \
    --prompt "Artificial Intelligence on Arch Linux is" \
    --max-length 100
```

---

## 5️⃣ Test the Base GPT-2 Model

### `run_gpt2.py`

Run text generation directly with the original GPT-2 model, without using the fine-tuned weights.

### Default Usage

```bash
python run_gpt2.py
```

### Custom Prompt

```bash
python run_gpt2.py \
    --prompt "Artificial Intelligence on Arch Linux is" \
    --max-length 50
```

This script can be useful for comparing the behavior of the base model against the fine-tuned version.

---

# 📊 Base Model vs Fine-Tuned Model

A typical workflow for evaluating the fine-tuning process is:

```text
               Same Prompt
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   🤖 Base GPT-2       🧠 Fine-Tuned GPT-2
          │                   │
          ▼                   ▼
     Generated Text      Generated Text
          │                   │
          └─────────┬─────────┘
                    ▼
               Comparison
```

Using the same prompt and generation settings makes qualitative comparison easier.

---

# 📦 Dependencies

The project is based primarily on:

* 🤗 **Transformers**
* 🔥 **PyTorch**
* 📄 PDF extraction libraries used by `pdf_to_txt.py`
* 🐍 Python standard library

Install the exact versions used by the project through:

```bash
pip install -r requirements.txt
```

For reproducibility, keep `requirements.txt` synchronized with the working environment.

---

# 🧪 Recommended Project Workflow

For a complete training session:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Extract PDF text
python pdf_to_txt.py -i document.pdf -o data.txt

# 3. Fine-tune GPT-2
python full_train.py \
    --data-file data.txt \
    --epochs 5 \
    --batch-size 2 \
    --grad-accum 4 \
    --learning-rate 5e-5 \
    --save-dir ./gpt2-full-model

# 4. Export latest checkpoint
python save_current.py \
    --checkpoints-dir ./gpt2-full-model \
    --output-dir ./my_full_gpt2

# 5. Test the fine-tuned model
python test_model.py \
    --model ./my_full_gpt2 \
    --prompt "Artificial Intelligence on Arch Linux is" \
    --max-length 100
```

---

# 🔍 Troubleshooting

### CUDA / GPU Issues

Check whether PyTorch can access your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A result of:

```text
True
```

indicates that CUDA is available to PyTorch.

### Out-of-Memory Errors

Try reducing:

```text
--batch-size
```

and increasing:

```text
--grad-accum
```

For example:

```bash
--batch-size 1 --grad-accum 8
```

### Dataset Problems

Inspect `data.txt` before starting training and verify that:

* The PDF extraction produced readable text.
* Unwanted headers, footers, or repeated page content have been removed.
* The dataset is not empty.
* The text encoding is valid UTF-8.

---

# 🔐 Data & Model Considerations

Before training on PDF content, make sure you have the necessary rights to use the source material.

Do not include sensitive, private, or confidential documents in a public repository.

---

# 📌 Notes

* The base GPT-2 model and the fine-tuned model are separate inference targets.
* Checkpoints should be preserved until the final model has been successfully tested.
* Tokenizer files should remain compatible with the model configuration.
* Training results depend heavily on dataset quality, preprocessing, hyperparameters, and available hardware.
* Fine-tuning on a small or highly repetitive dataset can cause overfitting or undesirable generation behavior.

---

# 🛣️ Future Improvements

Possible extensions for this project include:

```text
✅ PDF → TXT preprocessing improvements
✅ Automatic dataset validation
✅ Train/validation split
✅ Evaluation metrics
✅ Resume-from-checkpoint support
✅ TensorBoard logging
✅ Config files for training parameters
✅ Better text chunking and tokenization
✅ Automatic checkpoint cleanup
✅ Interactive inference mode
✅ Web/API interface for the fine-tuned model
```

---

# 📄 License

Add the project's license information here.

Example:

```text
MIT License
```

---

# 👨‍💻 Author

**Your Name**

```text
GPT-2 Custom Fine-Tuning & PDF Data Pipeline
```

---

⭐ **Star the repository if you find it useful.**

Contributions, issues, and improvements are welcome.

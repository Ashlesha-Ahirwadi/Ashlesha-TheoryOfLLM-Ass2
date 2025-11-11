# Assignment 2: Fine-Tuning Language Models with LoRA

## Overview

This project demonstrates fine-tuning a smaller language model using **LoRA (Low-Rank Adaptation)** with the PEFT library to improve performance on a question-answer classification task. The base model from HuggingFace is sub-optimal on some questions, allowing us to demonstrate the impact of fine-tuning.

---

## Repository Structure
```
Assignment_2/
├── data/
│ ├── train.jsonl # Training data for fine-tuning
│ ├── val.jsonl # Validation data for evaluation
│ └── unit_test.jsonl # Small subset for unit test
├── models/
│ └── lora_finetuned_model/ # Saved LoRA fine-tuned model
├── outputs/
│ ├── finetuned_val_outputs.json # Fine-tuned model outputs
│ ├── finetuned_val_metrics.json # Fine-tuned model metrics
│ ├── base_model_outputs.json # Base model outputs
│ ├── base_model_metrics.json # Base model metrics
│ └── unit_test_model/ # Unit test outputs and metrics
├── src/
│ ├── finetune_eval.py # Evaluate base & fine-tuned models
│ ├── dataset_prep.py # Data preprocessing script
│ ├── base_model_eval.py # Evaluate base model only
│ ├── finetune_lora.py # LoRA fine-tuning script
│ └── unit_test.py # Standalone unit test
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```
---

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Ashlesha-Ahirwadi/Ashlesha-TheoryOfLLM-Ass2.git
cd Assignment_2
```

2. Create and activate a conda environment:
```bash
conda create -n lora_env_1 python=3.12 -y
conda activate lora_env_1
```

3. Install dependencies:
```bash
pip install -r requirements.txt

```
---

## Data

Training and validation data in data/train.jsonl and data/val.jsonl.

Unit test data in data/unit_test.jsonl.

If custom data was generated, it can be downloaded from: Google Drive Link
.
```
## Data format (JSONL):

{
  "question": "Your question here",
  "answer": "Correct category/label here"
}

```
Labels: World, Sports, Business, Sci/Tech

## Base Model Evaluation

We used distilbert-base-uncased as the base model (approx. 66M parameters).

## Evaluation on reserved validation questions:

Metric	Result
Accuracy	0.25
Weighted F1	0.1

Sample outputs:

## Question	True Label	Base Model Prediction
Who won the World Cup in 2022?	Sports	Business
Latest advancements in AI research?	Sci/Tech	World
Stock market trends today?	Business	Sports

## Analysis:

Base model performs poorly on domain-specific questions (e.g., Sports, Business).

Misclassifications show that small base models are under-optimized for our dataset.

## LoRA Fine-Tuning with PEFT

We fine-tuned the same base model using LoRA:

LoRA configuration:
```
r = 8

lora_alpha = 16

target_modules = ["q_lin", "v_lin"]

lora_dropout = 0.1

Task type: SEQ_CLS
```


## Evaluation after LoRA fine-tuning:

Metric	Result
Accuracy	0.918
Weighted F1	0.918

## Sample outputs:

Question	True Label	Fine-Tuned Prediction
Who won the World Cup in 2022?	Sports	Sports
Latest advancements in AI research?	Sci/Tech	Sci/Tech
Stock market trends today?	Business	Business

## Analysis:

Fine-tuned model correctly predicts domain-specific questions that base model failed.

Accuracy increased from 0.25 → 0.918, F1-weighted from 0.1 → 0.918

LoRA allows efficient fine-tuning of low-rank adaptations instead of full model weights.

Qualitative inspection confirms improved contextual understanding and reasoning.

## Unit Test

A small unit test script ensures LoRA fine-tuning works without full training.

File: src/unit_test.py

Dataset: data/unit_test.jsonl (small subset)

Trains for 3 epochs only.

Compares base model vs fine-tuned model metrics.

## Run the unit test:
```
python src/unit_test.py
```

## Sample metrics from unit test:
```
Model	  Accuracy	Weighted F1
Base	   0.2	       0.13
LoRA       0.5	       0.45
```
Metrics are saved in: outputs/unit_test_model/unit_test_metrics.json

## Observations

Base model underperforms on specific tasks; LoRA fine-tuning significantly improves predictions.

LoRA allows parameter-efficient fine-tuning, saving GPU memory and compute time.

Outputs are stored in outputs/ for both base and fine-tuned models.

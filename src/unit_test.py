import os
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
import evaluate

# --- Config ---
DATA_FILE = "data/unit_test.jsonl"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "outputs/unit_test_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# --- Load data ---
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

texts = [d["question"] for d in data]
labels = [list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(d["answer"])] for d in data]

# --- Split data ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
    "test": Dataset.from_dict({"text": test_texts, "label": test_labels})
})

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized = dataset.map(preprocess, batched=True)

# --- Metrics ---
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1_weighted": f1_score["f1"]}

# ======================================================
# 1️⃣ BASE MODEL EVALUATION (NO FINE-TUNING)
# ======================================================
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

base_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "base"),
    per_device_eval_batch_size=2,
    report_to="none"
)

base_trainer = Trainer(
    model=base_model,
    args=base_args,
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n🚀 Evaluating Base Model...")
base_results = base_trainer.evaluate()
print("✅ Base Model Results:")
for k, v in base_results.items():
    if k.startswith("eval_"):
        print(f"{k}: {v:.4f}")

# ======================================================
# 2️⃣ LoRA FINE-TUNED MODEL TRAIN + EVAL
# ======================================================
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
lora_model = get_peft_model(base_model, lora_config)

train_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "lora"),
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    logging_steps=1,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=lora_model,
    args=train_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n🎯 Training and Evaluating LoRA Fine-Tuned Model...")
trainer.train()
lora_results = trainer.evaluate()

print("\n✅ LoRA Fine-Tuned Results:")
for k, v in lora_results.items():
    if k.startswith("eval_"):
        print(f"{k}: {v:.4f}")

# ======================================================
# 3️⃣ SAVE COMPARISON METRICS
# ======================================================
comparison = {
    "base_model": {
        "accuracy": base_results["eval_accuracy"],
        "f1_weighted": base_results["eval_f1_weighted"]
    },
    "lora_finetuned": {
        "accuracy": lora_results["eval_accuracy"],
        "f1_weighted": lora_results["eval_f1_weighted"]
    }
}

metrics_path = os.path.join(OUTPUT_DIR, "unit_test_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)
print(f"\n📊 Saved comparison metrics to {metrics_path}")

# Save model + tokenizer
lora_model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))

print("\n✅ Unit test completed successfully.")

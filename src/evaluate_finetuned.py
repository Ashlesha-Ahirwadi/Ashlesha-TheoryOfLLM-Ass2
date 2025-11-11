# src/finetune_eval.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import evaluate

# --- Config ---
VAL_PATH = "data/val.jsonl"  # validation set with questions + answers
MODEL_DIR = "models/lora_finetuned_model"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label map
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# --- Load validation data ---
with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = [json.loads(line) for line in f]

texts = [d["question"] for d in val_data]
labels = [INV_LABEL_MAP[d["answer"]] for d in val_data]

# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# --- Dataset wrapper ---
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = SimpleDataset(encodings, labels)

# --- Load LoRA fine-tuned model ---
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=4)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Run predictions ---
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels_batch = batch["labels"].to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_batch.cpu().tolist())

# --- Compute metrics ---
metric_acc = evaluate.load("accuracy")
results_acc = metric_acc.compute(predictions=all_preds, references=all_labels)
accuracy = results_acc["accuracy"]

f1_metric = evaluate.load("f1")
f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="weighted")["f1"]

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1-score: {f1:.4f}")

# --- Save detailed outputs ---
output_file = os.path.join(OUTPUT_DIR, "finetuned_val_outputs.json")
out_data = []
for i, text in enumerate(texts):
    out_data.append({
        "question": text,
        "true_label": LABEL_MAP[labels[i]],
        "pred_label": LABEL_MAP[all_preds[i]]
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"Saved detailed outputs to {output_file}")

# --- Save metrics ---
metrics_file = os.path.join(OUTPUT_DIR, "finetuned_metrics.json")
metrics_data = {
    "accuracy": accuracy,
    "f1_weighted": f1
}
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2, ensure_ascii=False)

print(f"Saved evaluation metrics to {metrics_file}")

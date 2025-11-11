# src/base_model_eval.py
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_path", type=str, default="data/val.jsonl")  # validation set with question+answer
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--out", type=str, default="outputs/base_model_outputs.json")
    p.add_argument("--metrics_out", type=str, default="outputs/base_model_metrics.json")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Load validation data
    with open(args.val_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    
    texts = [d["question"] for d in val_data]
    true_labels = [list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(d["answer"])] for d in val_data]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=4)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

    out = []
    preds = []
    for text in texts:
        pred_scores = pipe(text)[0]  # list of dicts with label/score
        pred_label = max(pred_scores, key=lambda x: x["score"])["label"]
        # Map HuggingFace labels like "LABEL_0" -> integer 0
        pred_idx = int(pred_label.split("_")[1])
        preds.append(pred_idx)

        formatted_scores = {f"LABEL_{i}": float(pred_scores[i]["score"]) for i in range(len(pred_scores))}
        out.append({"question": text, "raw": pred_scores, "scores": formatted_scores, "pred_label": LABEL_MAP[pred_idx]})

    # Save predictions
    import os
    os.makedirs("outputs", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Compute metrics
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    class_report = classification_report(true_labels, preds, target_names=LABEL_MAP.values(), output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": class_report
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved base outputs to {args.out}")
    print(f"Saved evaluation metrics to {args.metrics_out}")
    print(f"Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")

if __name__ == "__main__":
    main()

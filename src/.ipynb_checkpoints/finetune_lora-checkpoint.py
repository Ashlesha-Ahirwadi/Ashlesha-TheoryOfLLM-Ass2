import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
OUT_DIR = "models/lora_finetuned_model"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="data/train.jsonl")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default=OUT_DIR)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    return p.parse_args()

def load_qa_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    questions = [x["question"] for x in lines]
    labels = [list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(x["answer"])] for x in lines]
    return {"text": questions, "label": labels}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_qa_dataset(args.train_path)
    train_ds = Dataset.from_dict(raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(ex):
        tok = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=args.max_length)
        tok["label"] = ex["label"]
        return tok

    tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")

    data_collator = DataCollatorWithPadding(tokenizer)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=4)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    import evaluate
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("LoRA fine-tuned model saved to:", args.output_dir)

if __name__ == "__main__":
    main()

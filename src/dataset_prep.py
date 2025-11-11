# src/dataset_prep.py
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import random

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def make_qa_example(text, label):
    question = (
        "Classify this news article into one of these categories: World, Sports, Business, Sci/Tech.\n"
        f"Text: {text}"
    )
    answer = LABEL_MAP[int(label)]
    return {"question": question, "answer": answer}

def main():
    ds = load_dataset("ag_news")

    # --- Train set ---
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(ds["train"], desc="Writing train.jsonl"):
            qa = make_qa_example(ex["text"], ex["label"])
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # --- Validation set ---
    val_path = os.path.join(OUT_DIR, "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(ds["test"], desc="Writing val.jsonl"):
            qa = make_qa_example(ex["text"], ex["label"])
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # --- Unit test set (small sample from test) ---
    unit_test_path = os.path.join(OUT_DIR, "unit_test.jsonl")
    unit_test_samples = ds["test"].shuffle(seed=42).select(range(50))  # 100 examples
    with open(unit_test_path, "w", encoding="utf-8") as fout:
        for ex in unit_test_samples:
            qa = make_qa_example(ex["text"], ex["label"])
            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"Saved: {train_path}, {val_path}, and {unit_test_path}")

if __name__ == "__main__":
    main()

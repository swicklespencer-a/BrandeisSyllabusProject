"""
RoBERTa sentence classifier for syllabus line classification.

Each line is classified independently using roberta-base with a linear
classification head (13 classes: 12 schema labels + "O").

After finding the best hyperparameters, the script also saves:
  - [CLS] embeddings and logits for all splits  →  used by train_roberta_crf.py
  - The fine-tuned model                        →  models/roberta/

Hyperparameter grid (evaluated on dev macro F1):
  learning_rate  : [5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
  num_epochs     : [2, 3, 5, 7, 10]
  per_device_batch: [16, 32, 64]
  warmup_ratio   : [0.0, 0.06, 0.1]
  weight_decay   : [0.0, 0.01, 0.1]

Run on GPU (Colab recommended):
  python train_roberta.py

To run a quick smoke-test on CPU (small subset, fewer HP combos):
  python train_roberta.py --smoke-test
"""

import argparse
import itertools
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from evaluate import (
    LABELS,
    compute_metrics,
    load_jsonl,
    plot_confusion_matrix,
    print_results,
    save_results,
)

MODEL_NAME   = "roberta-base"
MAX_LENGTH   = 128
MODEL_DIR    = "models/roberta"
RESULTS_PATH = "results/roberta_results.json"
CONFUSION_PATH = "results/roberta_confusion_matrix.png"

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


# ── Dataset ────────────────────────────────────────────────────────────────────

class SyllabusDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.encodings = tokenizer(
            [r["text"] for r in records],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [LABEL2ID[r["label"]] for r in records], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── HuggingFace compute_metrics callback ──────────────────────────────────────

def hf_compute_metrics(eval_pred):
    """Called by Trainer after each eval step; returns macro F1 for early stopping."""
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, tokenizer, records, batch_size=64, device="cpu"):
    """
    Run the fine-tuned model over records and return:
      cls_embeddings : np.ndarray  [N, hidden_size]
      logits         : np.ndarray  [N, num_labels]
      true_labels    : list[str]
      doc_ids        : list[str]
    """
    model.eval()
    model.to(device)

    all_cls, all_logits, all_labels, all_doc_ids = [], [], [], []

    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        enc = tokenizer(
            [r["text"] for r in batch],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc, output_hidden_states=True)
        # [CLS] token is the first token of the last hidden state
        cls = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        logits = outputs.logits.cpu().numpy()

        all_cls.append(cls)
        all_logits.append(logits)
        all_labels.extend([r["label"] for r in batch])
        all_doc_ids.extend([r["doc_id"] for r in batch])

    return (
        np.vstack(all_cls),
        np.vstack(all_logits),
        all_labels,
        all_doc_ids,
    )


def save_embeddings(split_name, cls_emb, logits, labels, doc_ids):
    os.makedirs("data", exist_ok=True)
    np.save(f"data/cls_embeddings_{split_name}.npy", cls_emb)
    np.save(f"data/cls_logits_{split_name}.npy",     logits)
    np.save(f"data/cls_labels_{split_name}.npy",
            np.array([LABEL2ID[l] for l in labels]))
    with open(f"data/cls_docids_{split_name}.json", "w") as f:
        json.dump(doc_ids, f)
    print(f"  Embeddings saved: data/cls_*_{split_name}.*")


# ── Single training run ────────────────────────────────────────────────────────

def run_training(train_dataset, dev_dataset, lr, epochs, batch_size,
                 warmup_ratio, weight_decay, seed=42):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    # Use a temp directory for checkpoints so they are automatically cleaned up.
    # load_best_model_at_end=True needs checkpoints on disk during the run, but
    # we don't want to keep 225 × 10 epochs worth of 500MB RoBERTa checkpoints.
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,        # keep only the latest checkpoint → caps disk use to ~1 GB
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=seed,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=hf_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

        # Read best dev F1 before the temp dir is deleted
        best_f1 = max(
            (log["eval_macro_f1"] for log in trainer.state.log_history if "eval_macro_f1" in log),
            default=0.0,
        )
        best_model = trainer.model
    # tmp_dir and all checkpoints are deleted here automatically

    return best_model, best_f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Verify the script runs: 200/50/50 lines, 1 combo.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a reduced 12-combo grid on the full dataset for a fast performance check.")
    args = parser.parse_args()

    if args.smoke_test and args.quick:
        print("ERROR: --smoke-test and --quick are mutually exclusive.")
        sys.exit(1)

    print()
    print("=== ROBERTA SENTENCE CLASSIFIER ===")
    if args.quick:
        print("  [quick mode] Running reduced grid (12 combos) on full dataset.")
    elif args.smoke_test:
        print("  [smoke-test] Running 1 combo on 200/50/50 lines.")
    print()

    for path in ("data/train.jsonl", "data/dev.jsonl", "data/test.jsonl"):
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found.  Run split_data.py first.")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("  Loading data...")
    train_records = load_jsonl("data/train.jsonl")
    dev_records   = load_jsonl("data/dev.jsonl")
    test_records  = load_jsonl("data/test.jsonl")

    if args.smoke_test:
        print("  [smoke-test] Using 200 train / 50 dev / 50 test records.")
        train_records = train_records[:200]
        dev_records   = dev_records[:50]
        test_records  = test_records[:50]

    print(f"  Lines — train: {len(train_records)}  dev: {len(dev_records)}  test: {len(test_records)}")

    print(f"  Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("  Tokenizing datasets...")
    train_dataset = SyllabusDataset(train_records, tokenizer)
    dev_dataset   = SyllabusDataset(dev_records,   tokenizer)
    test_dataset  = SyllabusDataset(test_records,  tokenizer)

    # ── Hyperparameter grid ────────────────────────────────────────────────────
    if args.smoke_test:
        grid = {
            "lr":           [2e-5],
            "epochs":       [2],
            "batch_size":   [32],
            "warmup_ratio": [0.06],
            "weight_decay": [0.01],
        }
    elif args.quick:
        grid = {
            "lr":           [1e-5, 2e-5, 5e-5],
            "epochs":       [3, 5],
            "batch_size":   [32],
            "warmup_ratio": [0.06],
            "weight_decay": [0.01],
        }
    else:
        grid = {
            "lr":           [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
            "epochs":       [2, 3, 5, 7, 10],
            "batch_size":   [16, 32, 64],
            "warmup_ratio": [0.0, 0.06, 0.1],
            "weight_decay": [0.0, 0.01, 0.1],
        }

    combos = list(itertools.product(
        grid["lr"], grid["epochs"], grid["batch_size"],
        grid["warmup_ratio"], grid["weight_decay"],
    ))
    print(f"\n  Hyperparameter search: {len(combos)} combinations...\n")

    best_f1     = -1
    best_model  = None
    best_params = None
    search_log  = []

    for i, (lr, epochs, bs, wr, wd) in enumerate(combos, 1):
        print(f"  [{i:>3}/{len(combos)}] lr={lr}  epochs={epochs}  batch={bs}"
              f"  warmup={wr}  wd={wd}")

        model, dev_f1 = run_training(
            train_dataset, dev_dataset,
            lr=lr, epochs=epochs, batch_size=bs,
            warmup_ratio=wr, weight_decay=wd,
        )

        params = {"lr": lr, "epochs": epochs, "batch_size": bs,
                  "warmup_ratio": wr, "weight_decay": wd}
        search_log.append({**params, "dev_macro_f1": round(dev_f1, 4)})
        print(f"         dev macro F1: {dev_f1 * 100:.2f}")

        if dev_f1 > best_f1:
            best_f1     = dev_f1
            best_model  = model
            best_params = params
            print(f"         *** NEW BEST ***")

    print()
    print(f"  Best params:       {best_params}")
    print(f"  Best dev macro F1: {best_f1 * 100:.2f}")

    # ── Save the best model ────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)
    print(f"\n  Model saved to {MODEL_DIR}/")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    print("\n  Evaluating best model on test set...")
    best_model.eval()
    best_model.to(device)

    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=64)
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")
            outputs = best_model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels_batch.cpu().tolist())

    y_true = [ID2LABEL[i] for i in all_true]
    y_pred = [ID2LABEL[i] for i in all_preds]

    metrics = compute_metrics(y_true, y_pred, labels=LABELS)
    metrics["best_params"]   = best_params
    metrics["best_dev_f1"]   = round(best_f1, 4)
    metrics["hyperparameter_search"] = search_log

    print_results(metrics, model_name="RoBERTa (test set)")
    os.makedirs("results", exist_ok=True)
    save_results(metrics, RESULTS_PATH)
    plot_confusion_matrix(y_true, y_pred, LABELS, CONFUSION_PATH)

    # ── Extract and save embeddings for train_roberta_crf.py ──────────────────
    print("\n  Extracting [CLS] embeddings and logits for all splits...")
    for split_name, records in [("train", train_records),
                                 ("dev",   dev_records),
                                 ("test",  test_records)]:
        cls_emb, logits, labels, doc_ids = extract_embeddings(
            best_model, tokenizer, records, batch_size=64, device=device
        )
        save_embeddings(split_name, cls_emb, logits, labels, doc_ids)

    print()
    print("  Done.  Next step: python train_roberta_crf.py")
    print()


if __name__ == "__main__":
    main()

"""
RoBERTa + CRF model for syllabus sentence classification.

This is a two-stage model:
  Stage 1 (done): train_roberta.py fine-tuned RoBERTa and saved per-line
                  logit vectors (shape [N, 13]) for each split.
  Stage 2 (here): treat those logit vectors as CRF emission scores and learn
                  a transition matrix on top.  The CRF decodes each document's
                  label sequence jointly using Viterbi, capturing patterns like
                  "SCHEDULE usually follows SCHEDULE" that RoBERTa alone misses.

Architecture:
  For each document with N lines:
    emissions[i] = RoBERTa logit vector for line i   (shape [13])
    CRF learns P(y_1, ..., y_N | emissions) with a learned transition matrix
    Inference: Viterbi decoding over the full sequence

Implementation:
  Uses pytorch-crf (TorchCRF).  The RoBERTa weights are frozen — we only
  train the CRF transition matrix (13×13 = 169 parameters).

Hyperparameter grid:
  learning_rate : [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
  num_epochs    : [10, 20, 50, 100, 200]
  batch_size    : [8, 16, 32]   (number of documents per batch)
  optimizer     : ['adam', 'sgd', 'adamw']
  l2_reg        : [0.0, 1e-4, 1e-3, 1e-2]   (weight decay for transition matrix)

Output:
  models/roberta_crf_transitions.pt   (the learned CRF transition matrix)
  results/roberta_crf_results.json
  results/roberta_crf_confusion_matrix.png

Run (after train_roberta.py has completed):
  python train_roberta_crf.py
"""

import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

from evaluate import (
    LABELS,
    compute_metrics,
    plot_confusion_matrix,
    print_results,
    save_results,
)

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

MODEL_PATH   = "models/roberta_crf_transitions.pt"
RESULTS_PATH = "results/roberta_crf_results.json"
CONFUSION_PATH = "results/roberta_crf_confusion_matrix.png"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(split_name):
    """
    Load pre-computed RoBERTa logits and reconstruct document sequences.

    Returns:
      doc_sequences: list of tensors, each shape [doc_len, NUM_LABELS]
      doc_labels:    list of tensors, each shape [doc_len]  (int label ids)
    """
    logits_path  = f"data/cls_logits_{split_name}.npy"
    labels_path  = f"data/cls_labels_{split_name}.npy"
    docids_path  = f"data/cls_docids_{split_name}.json"

    for p in (logits_path, labels_path, docids_path):
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found.")
            print("  Run train_roberta.py first to generate embeddings.")
            sys.exit(1)

    logits  = np.load(logits_path)     # [N, NUM_LABELS]
    labels  = np.load(labels_path)     # [N]  (int ids)
    with open(docids_path) as f:
        doc_ids = json.load(f)         # [N]  (strings)

    # Group by doc_id, preserving the original order
    from collections import defaultdict, OrderedDict
    by_doc = OrderedDict()
    for i, doc_id in enumerate(doc_ids):
        if doc_id not in by_doc:
            by_doc[doc_id] = {"logits": [], "labels": []}
        by_doc[doc_id]["logits"].append(logits[i])
        by_doc[doc_id]["labels"].append(int(labels[i]))

    doc_sequences = [
        torch.tensor(np.array(v["logits"]), dtype=torch.float32)
        for v in by_doc.values()
    ]
    doc_labels = [
        torch.tensor(v["labels"], dtype=torch.long)
        for v in by_doc.values()
    ]
    return doc_sequences, doc_labels


# ── CRF wrapper ────────────────────────────────────────────────────────────────

class RobertaCRF(nn.Module):
    """
    Minimal wrapper: takes pre-computed emission scores and applies a CRF.
    The only learned parameters are the CRF transition matrix (13×13).
    """
    def __init__(self, num_labels):
        super().__init__()
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, emissions, labels=None, mask=None):
        """
        emissions: [batch, seq_len, num_labels]
        labels:    [batch, seq_len]  (required for training)
        mask:      [batch, seq_len] bool  (optional)
        Returns log-likelihood (scalar) if labels given, else Viterbi tags.
        """
        if labels is not None:
            return self.crf(emissions, labels, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)


# ── Batching ───────────────────────────────────────────────────────────────────

def make_batches(doc_sequences, doc_labels, batch_size):
    """
    Yield (emissions, labels, mask) tensors for each mini-batch.
    Documents in a batch are padded to the length of the longest document.
    """
    n = len(doc_sequences)
    indices = list(range(n))

    for start in range(0, n, batch_size):
        batch_idx = indices[start: start + batch_size]
        batch_seqs   = [doc_sequences[i] for i in batch_idx]
        batch_labels = [doc_labels[i]    for i in batch_idx]

        max_len = max(s.shape[0] for s in batch_seqs)
        B = len(batch_seqs)

        emissions = torch.zeros(B, max_len, NUM_LABELS)
        labels    = torch.zeros(B, max_len, dtype=torch.long)
        mask      = torch.zeros(B, max_len, dtype=torch.bool)

        for j, (seq, lbl) in enumerate(zip(batch_seqs, batch_labels)):
            L = seq.shape[0]
            emissions[j, :L] = seq
            labels[j, :L]    = lbl
            mask[j, :L]      = True

        yield emissions, labels, mask


# ── Training ───────────────────────────────────────────────────────────────────

def train_crf(train_seqs, train_lbls, dev_seqs, dev_lbls,
              lr, epochs, batch_size, optimizer_name, l2_reg,
              device):
    model = RobertaCRF(NUM_LABELS).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg,
                              momentum=0.9)

    best_dev_f1 = -1.0
    best_state  = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        # Shuffle document order each epoch
        perm = torch.randperm(len(train_seqs)).tolist()
        shuffled_seqs = [train_seqs[i] for i in perm]
        shuffled_lbls = [train_lbls[i] for i in perm]

        for emissions, labels, mask in make_batches(shuffled_seqs, shuffled_lbls, batch_size):
            emissions = emissions.to(device)
            labels    = labels.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            log_likelihood = model(emissions, labels, mask)
            loss = -log_likelihood         # maximize log-likelihood = minimize negative
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # Evaluate on dev
        dev_f1 = evaluate_f1(model, dev_seqs, dev_lbls, device)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"      epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  dev_f1={dev_f1*100:.2f}")

    model.load_state_dict(best_state)
    return model, best_dev_f1


@torch.no_grad()
def evaluate_f1(model, doc_seqs, doc_lbls, device):
    from sklearn.metrics import f1_score
    model.eval()
    y_true, y_pred = [], []

    for emissions, labels, mask in make_batches(doc_seqs, doc_lbls, batch_size=32):
        emissions = emissions.to(device)
        mask      = mask.to(device)

        pred_seqs = model(emissions, mask=mask)  # list of lists

        for j, (pred_seq, lbl_tensor, msk) in enumerate(
                zip(pred_seqs, labels, mask)):
            true_len = msk.sum().item()
            y_true.extend(lbl_tensor[:true_len].tolist())
            y_pred.extend(pred_seq[:true_len])

    non_o = [i for i, l in ID2LABEL.items() if l != "O"]
    return f1_score(y_true, y_pred, labels=non_o, average="macro", zero_division=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=== ROBERTA + CRF ===")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("  Loading pre-computed RoBERTa logits...")
    train_seqs, train_lbls = load_split("train")
    dev_seqs,   dev_lbls   = load_split("dev")
    test_seqs,  test_lbls  = load_split("test")
    print(f"  Docs — train: {len(train_seqs)}  dev: {len(dev_seqs)}  test: {len(test_seqs)}")

    # ── Hyperparameter grid ────────────────────────────────────────────────────
    grid = {
        "lr":             [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "epochs":         [10, 20, 50, 100, 200],
        "batch_size":     [8, 16, 32],
        "optimizer":      ["adam", "adamw", "sgd"],
        "l2_reg":         [0.0, 1e-4, 1e-3, 1e-2],
    }

    combos = list(itertools.product(
        grid["lr"], grid["epochs"], grid["batch_size"],
        grid["optimizer"], grid["l2_reg"],
    ))
    print(f"\n  Hyperparameter search: {len(combos)} combinations...\n")

    best_f1     = -1.0
    best_model  = None
    best_params = None
    search_log  = []

    for i, (lr, epochs, bs, opt, l2) in enumerate(combos, 1):
        print(f"  [{i:>4}/{len(combos)}] lr={lr}  epochs={epochs}  batch={bs}"
              f"  opt={opt}  l2={l2}")

        model, dev_f1 = train_crf(
            train_seqs, train_lbls,
            dev_seqs,   dev_lbls,
            lr=lr, epochs=epochs, batch_size=bs,
            optimizer_name=opt, l2_reg=l2,
            device=device,
        )

        params = {"lr": lr, "epochs": epochs, "batch_size": bs,
                  "optimizer": opt, "l2_reg": l2}
        search_log.append({**params, "dev_macro_f1": round(dev_f1, 4)})

        if dev_f1 > best_f1:
            best_f1     = dev_f1
            best_model  = model
            best_params = params
            print(f"  *** NEW BEST  dev macro F1: {dev_f1 * 100:.2f} ***")

    print()
    print(f"  Best params:       {best_params}")
    print(f"  Best dev macro F1: {best_f1 * 100:.2f}")

    # ── Evaluate on test ───────────────────────────────────────────────────────
    print("\n  Evaluating on test set...")
    best_model.eval()
    y_true_flat, y_pred_flat = [], []

    with torch.no_grad():
        for emissions, labels, mask in make_batches(test_seqs, test_lbls, batch_size=32):
            emissions = emissions.to(device)
            mask      = mask.to(device)
            pred_seqs = best_model(emissions, mask=mask)

            for j, (pred_seq, lbl_tensor, msk) in enumerate(
                    zip(pred_seqs, labels, mask)):
                true_len = msk.sum().item()
                y_true_flat.extend(lbl_tensor[:true_len].tolist())
                y_pred_flat.extend(pred_seq[:true_len])

    y_true_str = [ID2LABEL[i] for i in y_true_flat]
    y_pred_str = [ID2LABEL[i] for i in y_pred_flat]

    metrics = compute_metrics(y_true_str, y_pred_str, labels=LABELS)
    metrics["best_params"]   = best_params
    metrics["best_dev_f1"]   = round(best_f1, 4)
    metrics["hyperparameter_search"] = search_log

    print_results(metrics, model_name="RoBERTa + CRF (test set)")

    # Save
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(best_model.state_dict(), MODEL_PATH)
    print(f"  CRF transition matrix saved to {MODEL_PATH}")

    save_results(metrics, RESULTS_PATH)
    plot_confusion_matrix(y_true_str, y_pred_str, LABELS, CONFUSION_PATH)

    print()
    print("  All three models complete.  Results in results/")
    print()


if __name__ == "__main__":
    main()

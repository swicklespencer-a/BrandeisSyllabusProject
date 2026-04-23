"""
CRF baseline for syllabus sentence classification.

The CRF operates on sequences of lines within each document, which lets it
learn label transition patterns (e.g. SCHEDULE usually follows SCHEDULE,
INTEGRITY rarely follows GRADE).

Features per line:
  - Bag-of-words (top 5000 unigrams, binary)
  - Normalized position in document
  - Line length bucket
  - Header heuristic (all-caps or ends with ":")
  - Keyword groups for each label category
  - Window features: same features from the previous and next line
    (prev_ / next_ prefixed, so the CRF sees local context)

Hyperparameter grid (full search on dev macro F1):
  c1            : [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
  c2            : [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
  max_iterations: [100, 200, 500, 1000]
  algorithm     : ['lbfgs', 'l2sgd', 'pa']

Note: 'l2sgd' ignores c1; 'pa' ignores both c1 and c2.

Output:
  models/crf_model.pkl
  results/crf_results.json
  results/crf_confusion_matrix.png

Run:
  python train_crf.py              # full grid (432 combos)
  python train_crf.py --quick      # reduced grid (18 combos) for a fast sanity check
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import joblib
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_f1_score

from evaluate import (
    LABELS,
    compute_metrics,
    plot_confusion_matrix,
    print_results,
    save_results,
    load_jsonl,
)

MODEL_PATH = "models/crf_model.pkl"
RESULTS_PATH = "results/crf_results.json"
CONFUSION_PATH = "results/crf_confusion_matrix.png"

# ── Keyword groups ─────────────────────────────────────────────────────────────
KEYWORDS = {
    "grade":     {"grade", "grading", "graded", "point", "points", "percent",
                  "%", "score", "gpa", "credit", "pass", "fail", "rubric"},
    "schedule":  {"week", "date", "due", "monday", "tuesday", "wednesday",
                  "thursday", "friday", "january", "february", "march", "april",
                  "may", "june", "july", "august", "september", "october",
                  "november", "december", "deadline", "session", "lecture",
                  "class", "meeting", "calendar"},
    "integrity": {"plagiarism", "plagiarize", "cheat", "cheating", "dishonesty",
                  "integrity", "academic", "misconduct", "citation", "cite",
                  "honor", "turnitin"},
    "late":      {"late", "penalty", "extension", "overdue", "makeup",
                  "make-up", "missed", "incomplete"},
    "attend":    {"attendance", "attend", "absent", "absence", "present",
                  "participation", "tardy", "excused", "unexcused"},
    "assign":    {"assignment", "homework", "hw", "problem", "exercise",
                  "project", "submit", "submission", "canvas", "blackboard",
                  "upload", "paper", "essay", "report"},
    "material":  {"textbook", "reading", "text", "book", "article", "pdf",
                  "chapter", "resource", "required", "recommended", "isbn",
                  "library", "reserve"},
    "accom":     {"accommodation", "disability", "accessible", "accessibility",
                  "services", "ada", "support", "special", "need", "request"},
    "conduct":   {"conduct", "behavior", "respect", "policy", "harassment",
                  "discrimination", "professional", "classroom", "cell phone",
                  "device", "laptop"},
    "admin":     {"office", "hours", "email", "contact", "instructor",
                  "professor", "ta", "teaching assistant", "syllabus",
                  "prerequisite", "credit", "department"},
    "descrip":   {"course", "overview", "objective", "goal", "learn",
                  "learning", "outcome", "description", "topic", "cover",
                  "introduction", "focus", "aim"},
    "ctf":       {"ctf", "capture", "flag", "challenge", "competition",
                  "hack", "hacking", "security", "cyber"},
}


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_vocab(sequences, top_n=5000):
    counts = Counter()
    for seq in sequences:
        for line in seq:
            for word in line["text"].lower().split():
                counts[word] += 1
    return {word for word, _ in counts.most_common(top_n)}


# ── Feature extraction ─────────────────────────────────────────────────────────

def line_features(text, position, doc_len, vocab):
    """Return a feature dict for a single line."""
    words = text.lower().split()
    word_set = set(words)

    feats = {}

    # Bag-of-words (binary, vocabulary-filtered)
    for w in word_set:
        if w in vocab:
            feats[f"w:{w}"] = True

    # Position features
    feats["position"] = round(position / max(doc_len - 1, 1), 3)
    feats["pos_bucket"] = str(min(int(position / max(doc_len, 1) * 10), 9))  # 0-9
    feats["is_first"] = position == 0
    feats["is_last"]  = position == doc_len - 1

    # Length features
    n_chars = len(text)
    feats["len_chars"] = min(n_chars, 500)  # cap to avoid outlier dominance
    feats["len_words"] = len(words)
    feats["is_short"]  = n_chars < 15
    feats["is_long"]   = n_chars > 200

    # Header heuristic
    stripped = text.strip()
    feats["is_all_caps"]    = stripped.isupper() and len(stripped) > 2
    feats["ends_colon"]     = stripped.endswith(":")
    feats["starts_number"]  = stripped[:1].isdigit()
    feats["starts_bullet"]  = stripped[:1] in "-•*"

    # Keyword groups
    for group, kws in KEYWORDS.items():
        feats[f"kw:{group}"] = bool(word_set & kws)

    return feats


def doc_to_feature_sequence(doc_lines, vocab):
    """
    Convert a list of line dicts (one document) to a list of feature dicts,
    including window features from the previous and next lines.
    """
    n = len(doc_lines)
    base = [
        line_features(doc_lines[i]["text"], i, n, vocab)
        for i in range(n)
    ]

    result = []
    for i, feats in enumerate(base):
        combined = dict(feats)

        # Previous line features
        if i > 0:
            for k, v in base[i - 1].items():
                combined[f"prev_{k}"] = v
        else:
            combined["BOS"] = True  # beginning of sequence

        # Next line features
        if i < n - 1:
            for k, v in base[i + 1].items():
                combined[f"next_{k}"] = v
        else:
            combined["EOS"] = True  # end of sequence

        result.append(combined)

    return result


# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(path):
    """Return (sequences, label_sequences) grouped by doc_id."""
    records = load_jsonl(path)
    by_doc = defaultdict(list)
    for rec in records:
        by_doc[rec["doc_id"]].append(rec)

    # Sort lines within each doc by line_idx to preserve document order
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda r: r["line_idx"])

    sequences = list(by_doc.values())
    label_seqs = [[r["label"] for r in seq] for seq in sequences]
    return sequences, label_seqs


# ── Training ───────────────────────────────────────────────────────────────────

def train_and_eval(X_train, y_train, X_dev, y_dev, algorithm, c1, c2, max_iter):
    """Train one CRF and return dev macro F1."""
    kwargs = dict(
        algorithm=algorithm,
        max_iterations=max_iter,
        all_possible_transitions=True,
    )
    if algorithm == "lbfgs":
        kwargs["c1"] = c1
        kwargs["c2"] = c2
    elif algorithm == "l2sgd":
        kwargs["c2"] = c2  # l2sgd uses c2 only
    # 'pa' uses neither

    crf = sklearn_crfsuite.CRF(**kwargs)
    try:
        crf.fit(X_train, y_train)
    except Exception as e:
        return None, None, str(e)

    f1 = flat_f1_score(
        y_dev, crf.predict(X_dev),
        average="macro",
        labels=[l for l in LABELS if l != "O"],  # macro over non-O labels
        zero_division=0,
    )
    return crf, f1, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run a reduced 18-combo grid for a fast performance check.")
    args = parser.parse_args()

    print()
    print("=== CRF BASELINE TRAINING ===")
    if args.quick:
        print("  [quick mode] Running reduced grid (18 combos).")
    print()

    for path in ("data/train.jsonl", "data/dev.jsonl", "data/test.jsonl"):
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found.  Run split_data.py first.")
            sys.exit(1)

    print("  Loading data...")
    train_seqs, train_labels = load_split("data/train.jsonl")
    dev_seqs,   dev_labels   = load_split("data/dev.jsonl")
    test_seqs,  test_labels  = load_split("data/test.jsonl")
    print(f"  Docs  — train: {len(train_seqs)}  dev: {len(dev_seqs)}  test: {len(test_seqs)}")
    print(f"  Lines — train: {sum(len(s) for s in train_seqs)}"
          f"  dev: {sum(len(s) for s in dev_seqs)}"
          f"  test: {sum(len(s) for s in test_seqs)}")

    print("  Building vocabulary (top 5000 words from train)...")
    vocab = build_vocab(train_seqs, top_n=5000)
    print(f"  Vocabulary size: {len(vocab)}")

    print("  Extracting features...")
    X_train = [doc_to_feature_sequence(seq, vocab) for seq in train_seqs]
    X_dev   = [doc_to_feature_sequence(seq, vocab) for seq in dev_seqs]
    X_test  = [doc_to_feature_sequence(seq, vocab) for seq in test_seqs]

    # ── Hyperparameter grid search ─────────────────────────────────────────────
    if args.quick:
        grid = {
            "c1":             [0.01, 0.1, 1.0],
            "c2":             [0.01, 0.1, 1.0],
            "max_iterations": [200],
            "algorithm":      ["lbfgs", "pa"],
        }
    else:
        grid = {
            "c1":             [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
            "c2":             [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
            "max_iterations": [100, 200, 500, 1000],
            "algorithm":      ["lbfgs", "l2sgd", "pa"],
        }

    combos = list(itertools.product(
        grid["algorithm"],
        grid["c1"],
        grid["c2"],
        grid["max_iterations"],
    ))
    total = len(combos)
    print(f"\n  Hyperparameter search: {total} combinations...")
    print()

    best_f1    = -1
    best_crf   = None
    best_params = None
    search_log  = []

    for i, (algo, c1, c2, max_iter) in enumerate(combos, 1):
        crf, f1, err = train_and_eval(
            X_train, train_labels,
            X_dev,   dev_labels,
            algo, c1, c2, max_iter,
        )
        if err:
            continue

        search_log.append({"algorithm": algo, "c1": c1, "c2": c2,
                            "max_iterations": max_iter, "dev_macro_f1": round(f1, 4)})

        if f1 > best_f1:
            best_f1     = f1
            best_crf    = crf
            best_params = {"algorithm": algo, "c1": c1, "c2": c2, "max_iterations": max_iter}
            print(f"  [{i:>4}/{total}] NEW BEST  algo={algo}  c1={c1}  c2={c2}"
                  f"  max_iter={max_iter}  dev_macro_f1={f1 * 100:.2f}")
        elif i % 50 == 0:
            print(f"  [{i:>4}/{total}] best so far: {best_f1 * 100:.2f}  "
                  f"(algo={best_params['algorithm']})")

    print()
    print(f"  Best hyperparameters: {best_params}")
    print(f"  Best dev macro F1:    {best_f1 * 100:.2f}")

    # ── Evaluate best model on test ────────────────────────────────────────────
    print()
    print("  Evaluating best model on test set...")
    y_pred_seqs = best_crf.predict(X_test)

    y_true_flat = [lbl for seq in test_labels   for lbl in seq]
    y_pred_flat = [lbl for seq in y_pred_seqs   for lbl in seq]

    metrics = compute_metrics(y_true_flat, y_pred_flat, labels=LABELS)
    metrics["best_params"]   = best_params
    metrics["best_dev_f1"]   = round(best_f1, 4)
    metrics["hyperparameter_search"] = search_log

    print_results(metrics, model_name="CRF (test set)")

    # Save model and results
    os.makedirs("models",   exist_ok=True)
    os.makedirs("results",  exist_ok=True)
    joblib.dump({"crf": best_crf, "vocab": vocab}, MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")

    save_results(metrics, RESULTS_PATH)
    plot_confusion_matrix(y_true_flat, y_pred_flat, LABELS, CONFUSION_PATH)


if __name__ == "__main__":
    main()

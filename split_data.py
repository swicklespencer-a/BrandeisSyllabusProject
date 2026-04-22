"""
Split data/sentences.jsonl into train / dev / test at the document level.

Splitting at the document level ensures that all lines from a given syllabus
end up in exactly one split, preventing any data leakage between splits.

Default split ratios: 80% train / 10% dev / 10% test.

Output:
  data/train.jsonl
  data/dev.jsonl
  data/test.jsonl

Run:
  python split_data.py
  python split_data.py --train 0.7 --dev 0.15 --test 0.15   # custom ratios
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

INPUT_FILE = "data/sentences.jsonl"
SEED = 42


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_split_stats(name, records, doc_ids):
    label_counts = Counter(r["label"] for r in records)
    total = sum(label_counts.values())
    print(f"  {name}: {len(doc_ids)} docs, {len(records)} lines")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"    {label:<12} {count:>6}  ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Split sentences.jsonl into train/dev/test.")
    parser.add_argument("--train", type=float, default=0.80, help="Train fraction (default 0.80)")
    parser.add_argument("--dev",   type=float, default=0.10, help="Dev fraction (default 0.10)")
    parser.add_argument("--test",  type=float, default=0.10, help="Test fraction (default 0.10)")
    parser.add_argument("--seed",  type=int,   default=SEED,  help="Random seed (default 42)")
    args = parser.parse_args()

    total_frac = args.train + args.dev + args.test
    if abs(total_frac - 1.0) > 1e-6:
        print(f"  ERROR: train + dev + test must sum to 1.0 (got {total_frac:.4f})")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: {INPUT_FILE} not found.")
        print("  Run 'python convert_to_sentences.py' first.")
        sys.exit(1)

    print()
    print("=== SPLIT DATA ===")
    print()

    records = load_jsonl(INPUT_FILE)
    print(f"  Loaded {len(records)} lines from {INPUT_FILE}")

    # Group lines by doc_id
    by_doc = defaultdict(list)
    for rec in records:
        by_doc[rec["doc_id"]].append(rec)

    doc_ids = sorted(by_doc.keys())
    print(f"  Total documents: {len(doc_ids)}")

    # Shuffle at document level with fixed seed
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * args.train)
    n_dev   = int(n * args.dev)
    # test gets whatever is left so rounding doesn't drop any docs
    n_test  = n - n_train - n_dev

    train_ids = set(doc_ids[:n_train])
    dev_ids   = set(doc_ids[n_train:n_train + n_dev])
    test_ids  = set(doc_ids[n_train + n_dev:])

    train_records = [r for r in records if r["doc_id"] in train_ids]
    dev_records   = [r for r in records if r["doc_id"] in dev_ids]
    test_records  = [r for r in records if r["doc_id"] in test_ids]

    # Sanity check: no doc_id in multiple splits
    assert train_ids.isdisjoint(dev_ids),  "BUG: doc_id overlap between train and dev"
    assert train_ids.isdisjoint(test_ids), "BUG: doc_id overlap between train and test"
    assert dev_ids.isdisjoint(test_ids),   "BUG: doc_id overlap between dev and test"
    assert len(train_ids) + len(dev_ids) + len(test_ids) == n, "BUG: doc count mismatch"

    write_jsonl(train_records, "data/train.jsonl")
    write_jsonl(dev_records,   "data/dev.jsonl")
    write_jsonl(test_records,  "data/test.jsonl")

    print()
    print(f"  Split ratios: train={args.train:.0%}  dev={args.dev:.0%}  test={args.test:.0%}  (seed={args.seed})")
    print()
    print_split_stats("Train", train_records, train_ids)
    print()
    print_split_stats("Dev  ", dev_records,   dev_ids)
    print()
    print_split_stats("Test ", test_records,  test_ids)
    print()
    print("  Files written:")
    print("    data/train.jsonl")
    print("    data/dev.jsonl")
    print("    data/test.jsonl")
    print()


if __name__ == "__main__":
    main()

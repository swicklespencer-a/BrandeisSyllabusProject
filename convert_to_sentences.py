"""
Convert Label Studio annotations to sentence-level (line-level) JSONL.

Input:
  - label_studio_tasks.json     raw syllabus texts, 932 docs
  - annotations/all_annotations.json   span annotations from annotate.py

Output:
  - data/sentences.jsonl   one JSON object per non-empty line, with label

Each output record:
  {
    "doc_id":     str,
    "line_idx":   int,   # 0-based index of this line within the document
    "text":       str,
    "label":      str,   # one of the 12 schema labels, or "O"
    "char_start": int,   # character offset of line start in raw text
    "char_end":   int    # character offset of line end (exclusive)
  }

Label assignment:
  A line receives the label of any span that covers >50% of the line's
  characters.  If multiple spans qualify (shouldn't happen with clean
  annotations), the longest overlap wins.  If no span qualifies → "O".

Run:
  python convert_to_sentences.py
"""

import json
import os
import sys
from collections import Counter

TASKS_FILE = "label_studio_tasks.json"
ANNOTATIONS_FILE = "annotations/all_annotations.json"
OUTPUT_FILE = "data/sentences.jsonl"


def load_tasks(path):
    """Return {doc_id: {"text": ..., ...}} from label_studio_tasks.json."""
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return {t["data"]["doc_id"]: t["data"] for t in tasks}


def load_annotations(path):
    """
    Return a dict {doc_id: annotation} keeping only the last annotation per
    doc_id (in case the same doc was annotated multiple times).  Entries with
    validity == "Invalid" are dropped.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    by_doc = {}
    skipped_invalid = 0
    for entry in raw:
        if entry.get("validity") == "Invalid":
            skipped_invalid += 1
            continue
        doc_id = entry["doc_id"]
        by_doc[doc_id] = entry  # last one wins if duplicates exist

    if skipped_invalid:
        print(f"  Skipped {skipped_invalid} annotations marked Invalid.")
    return by_doc


def get_line_offsets(text):
    """
    Split text on '\\n' and return a list of (char_start, char_end, line_text)
    for every non-empty line (after stripping).

    char_start / char_end are offsets into the original text string.
    """
    lines = []
    pos = 0
    for raw_line in text.split("\n"):
        line_start = pos
        line_end = pos + len(raw_line)
        stripped = raw_line.strip()
        if stripped:
            lines.append((line_start, line_end, stripped))
        pos = line_end + 1  # +1 for the '\n' character itself
    return lines


def assign_label(line_start, line_end, spans):
    """
    Given a line's character range [line_start, line_end) and a list of span
    dicts (each with 'start', 'end', 'label'), return the label of the span
    whose overlap with this line exceeds 50% of the line's length.

    If no span qualifies, return "O".
    If multiple qualify, return the one with the greatest overlap.
    """
    line_len = line_end - line_start
    if line_len == 0:
        return "O"

    best_label = "O"
    best_overlap = 0

    for span in spans:
        overlap = min(line_end, span["end"]) - max(line_start, span["start"])
        if overlap > 0 and overlap / line_len > 0.5 and overlap > best_overlap:
            best_overlap = overlap
            best_label = span["label"]

    return best_label


def convert(tasks, annotations):
    """
    Iterate over annotated docs and convert to sentence-level records.
    Returns a list of dicts and a label Counter for reporting.
    """
    records = []
    label_counts = Counter()
    missing_text = 0

    for doc_id, ann in annotations.items():
        if doc_id not in tasks:
            missing_text += 1
            continue

        text = tasks[doc_id].get("text", "")
        spans = ann.get("spans", [])

        lines = get_line_offsets(text)
        for idx, (char_start, char_end, line_text) in enumerate(lines):
            label = assign_label(char_start, char_end, spans)
            label_counts[label] += 1
            records.append({
                "doc_id": doc_id,
                "line_idx": idx,
                "text": line_text,
                "label": label,
                "char_start": char_start,
                "char_end": char_end,
            })

    if missing_text:
        print(f"  WARNING: {missing_text} annotated docs not found in tasks file.")

    return records, label_counts


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print()
    print("=== CONVERT ANNOTATIONS TO SENTENCE-LEVEL JSONL ===")
    print()

    if not os.path.exists(TASKS_FILE):
        print(f"  ERROR: {TASKS_FILE} not found.")
        sys.exit(1)
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"  ERROR: {ANNOTATIONS_FILE} not found.")
        print("  Run 'python annotate.py finish' first to export annotations.")
        sys.exit(1)

    print(f"  Loading tasks from {TASKS_FILE}...")
    tasks = load_tasks(TASKS_FILE)
    print(f"  Loaded {len(tasks)} syllabi.")

    print(f"  Loading annotations from {ANNOTATIONS_FILE}...")
    annotations = load_annotations(ANNOTATIONS_FILE)
    print(f"  Loaded annotations for {len(annotations)} syllabi.")

    print("  Converting to line-level records...")
    records, label_counts = convert(tasks, annotations)

    write_jsonl(records, OUTPUT_FILE)

    print()
    print(f"  Total lines written: {len(records)}")
    print(f"  Output: {OUTPUT_FILE}")
    print()
    print("  Label distribution:")
    total = sum(label_counts.values())
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"    {label:<12} {count:>6}  ({pct:5.1f}%)")
    print()

    # Spot-check: print 3 labeled examples for manual verification
    print("  Sample labeled lines (spot-check these against Label Studio):")
    shown = 0
    for rec in records:
        if rec["label"] != "O" and shown < 3:
            print(f"    [{rec['label']}] \"{rec['text'][:80]}\"")
            shown += 1
    print()


if __name__ == "__main__":
    main()

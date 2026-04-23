"""
processing.py

Usage:
    python processing.py            # runs dedup, then split
    python processing.py dedup
    python processing.py split

Operates on annotations/all_annotations.json
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ANNOTATIONS_PATH = Path("annotations/all_annotations.json")
SPLIT_DIR = Path("annotations/split_annotations")


def _comparable(entry):
    """Return a copy of the entry without 'logged_at' for equality checks."""
    return {k: v for k, v in entry.items() if k != "logged_at"}


def _entry_key(entry):
    """Hashable representation of an entry, ignoring logged_at."""
    # json.dumps with sort_keys handles nested lists/dicts deterministically
    return json.dumps(_comparable(entry), sort_keys=True)


def dedup(path: Path = ANNOTATIONS_PATH):
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {path}")

    seen_keys = {}            # full-content key  -> first index kept
    by_doc_annotator = {}     # (doc_id, annotator) -> first kept entry
    by_doc_id = {}            # doc_id -> list of (annotator, entry)

    kept = []
    exact_dupes = 0
    conflict_same_annotator = []   # same doc_id + annotator, different content
    conflict_diff_annotator = []   # same doc_id, different annotator

    for entry in entries:
        key = _entry_key(entry)
        doc_id = entry.get("doc_id")
        annotator = entry.get("annotator")

        # 1. Exact duplicate (ignoring logged_at) -> drop
        if key in seen_keys:
            exact_dupes += 1
            continue

        # 2. Same doc_id + annotator but different content -> note
        da_key = (doc_id, annotator)
        if da_key in by_doc_annotator:
            conflict_same_annotator.append(
                (by_doc_annotator[da_key], entry)
            )

        # 3. Same doc_id but different annotator -> note
        if doc_id in by_doc_id:
            for prev_annotator, prev_entry in by_doc_id[doc_id]:
                if prev_annotator != annotator:
                    conflict_diff_annotator.append((prev_entry, entry))

        # Record and keep
        seen_keys[key] = len(kept)
        by_doc_annotator.setdefault(da_key, entry)
        by_doc_id.setdefault(doc_id, []).append((annotator, entry))
        kept.append(entry)

    # Write deduped file back
    with path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n--- Dedup report ---")
    print(f"Exact duplicates removed (same content, ignoring logged_at): {exact_dupes}")
    print(f"Entries remaining: {len(kept)}")
    print(f"Wrote deduped data back to {path}")

    if conflict_same_annotator:
        print(f"\n[!] {len(conflict_same_annotator)} case(s) where the SAME annotator "
              f"annotated the same doc_id with DIFFERENT content:")
        for a, b in conflict_same_annotator:
            print(f"  - doc_id={a.get('doc_id')} annotator={a.get('annotator')}")
            print(f"      first : rating={a.get('power_rating')} logged_at={a.get('logged_at')}")
            print(f"      second: rating={b.get('power_rating')} logged_at={b.get('logged_at')}")

    if conflict_diff_annotator:
        print(f"\n[i] {len(conflict_diff_annotator)} case(s) where the same doc_id "
              f"was annotated by DIFFERENT annotators (this may be expected):")
        for a, b in conflict_diff_annotator:
            print(f"  - doc_id={a.get('doc_id')}: "
                  f"{a.get('annotator')} vs {b.get('annotator')}")


def split(path: Path = ANNOTATIONS_PATH, out_dir: Path = SPLIT_DIR):
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    # Wipe the output directory completely so results are always fresh
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    by_annotator = defaultdict(list)
    for entry in entries:
        annotator = entry.get("annotator") or "unknown"
        by_annotator[annotator].append(entry)

    print(f"\n--- Split report ---")
    for annotator, items in sorted(by_annotator.items()):
        out_path = out_dir / f"{annotator}_annotations.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"  {out_path}: {len(items)} entries")
    print(f"Wrote {len(by_annotator)} annotator file(s) to {out_dir}")


def main():
    if len(sys.argv) < 2:
        # No command -> dedup, then split
        dedup()
        split()
        return

    command = sys.argv[1]
    if command == "dedup":
        dedup()
    elif command == "split":
        split()
    else:
        print(f"Unknown command: {command}")
        print("Commands: dedup, split  (or no command to run both)")
        sys.exit(1)


if __name__ == "__main__":
    main()
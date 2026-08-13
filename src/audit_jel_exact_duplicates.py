"""Report normalized exact duplicates in the raw JEL-training extract.

Duplicates are identified from case-folded title + abstract with collapsed
whitespace.  The source JSONL is never altered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "repec_jel_exact_duplicate_audit.json"
WHITESPACE = re.compile(r"\s+")


def fingerprint(row: dict[str, object]) -> str:
    """Return a stable fingerprint without retaining the full text in memory."""
    text = f"{row['title']}\n{row['abstract']}"
    normalised = WHITESPACE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def compact_record(row: dict[str, object]) -> dict[str, object]:
    return {
        "pid": row["pid"],
        "handle": row["handle"],
        "year": row["year"],
        "split": row["split"],
        "labels_1digit": row["labels_1digit"],
        "labels_2digit": row["labels_2digit"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw extract not found: {args.input}")

    # Each item starts as one compact record. It is expanded only if a second
    # occurrence is found, keeping memory use bounded for mostly-unique text.
    seen: dict[str, dict[str, object]] = {}
    record_count = 0
    for line in args.input.open(encoding="utf-8"):
        row = json.loads(line)
        record_count += 1
        key = fingerprint(row)
        record = compact_record(row)
        previous = seen.get(key)
        if previous is None:
            seen[key] = {
                "count": 1,
                "records": [record],
                "splits": {str(record["split"])},
                "label_signatures": {tuple(record["labels_2digit"])},
            }
            continue

        previous["count"] = int(previous["count"]) + 1
        splits = previous["splits"]
        label_signatures = previous["label_signatures"]
        assert isinstance(splits, set)
        assert isinstance(label_signatures, set)
        splits.add(str(record["split"]))
        label_signatures.add(tuple(record["labels_2digit"]))
        records = previous["records"]
        assert isinstance(records, list)
        if len(records) < 5:
            records.append(record)

    duplicate_groups = [item for item in seen.values() if int(item["count"]) > 1]
    duplicate_records = sum(int(item["count"]) for item in duplicate_groups)
    cross_split_groups = []
    conflicting_label_groups = []
    for item in duplicate_groups:
        records = item["records"]
        assert isinstance(records, list)
        splits = item["splits"]
        labels = item["label_signatures"]
        assert isinstance(splits, set)
        assert isinstance(labels, set)
        if len(splits) > 1:
            cross_split_groups.append(item)
        if len(labels) > 1:
            conflicting_label_groups.append(item)

    def example(item: dict[str, object]) -> dict[str, object]:
        records = item["records"]
        assert isinstance(records, list)
        return {
            "count": item["count"],
            "splits": sorted(item["splits"]),
            "records": records,
        }

    report = {
        "input": str(args.input),
        "records": record_count,
        "unique_normalized_texts": len(seen),
        "duplicate_groups": len(duplicate_groups),
        "records_in_duplicate_groups": duplicate_records,
        "additional_duplicate_records": duplicate_records - len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "duplicate_groups_with_conflicting_2digit_labels": len(conflicting_label_groups),
        "largest_duplicate_groups": [
            example(item)
            for item in sorted(duplicate_groups, key=lambda item: int(item["count"]), reverse=True)
            [: args.examples]
        ],
        "cross_split_examples": [example(item) for item in cross_split_groups[: args.examples]],
        "conflicting_label_examples": [
            example(item) for item in conflicting_label_groups[: args.examples]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote duplicate audit to {args.output}")
    print(f"Duplicate groups: {len(duplicate_groups):,}")
    print(f"Cross-split duplicate groups: {len(cross_split_groups):,}")


if __name__ == "__main__":
    main()

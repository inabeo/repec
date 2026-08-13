"""Audit unusually high JEL label counts without automatically removing rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "repec_jel_label_cardinality_v2.json"
SPLITS = ("train", "validation", "test", "holdout")
THRESHOLDS = (8, 10, 12, 15, 20)
REVIEW_THRESHOLD = 15
EXAMPLE_LIMIT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percentile(values: list[int], probability: float) -> int:
    values.sort()
    return values[round((len(values) - 1) * probability)]


def archive_from_handle(handle: str) -> str:
    parts = handle.split(":")
    return parts[1].casefold() if len(parts) > 1 else "unknown"


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Clean dataset not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    cardinalities: list[int] = []
    distribution = Counter()
    counts_by_split: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    high_by_archive = Counter()
    high_examples: list[dict[str, object]] = []
    invalid_rows = 0
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            labels = row.get("labels_2digit")
            if not isinstance(labels, list) or not labels:
                invalid_rows += 1
                continue
            count = len(labels)
            rows += 1
            cardinalities.append(count)
            distribution[count] += 1
            split = str(row["split"])
            for threshold in THRESHOLDS:
                if count >= threshold:
                    counts_by_split[split][threshold] += 1
            if count >= REVIEW_THRESHOLD:
                high_by_archive[archive_from_handle(str(row["handle"]))] += 1
                high_examples.append(
                    {
                        "pid": row["pid"],
                        "handle": row["handle"],
                        "year": row["year"],
                        "split": split,
                        "title": row["title"],
                        "label_count_2digit": count,
                        "labels_2digit": labels,
                    }
                )

    high_examples.sort(key=lambda row: (-int(row["label_count_2digit"]), int(row["year"]), int(row["pid"])))
    mean = sum(cardinalities) / len(cardinalities)
    reviewed_rows = sum(count for cardinality, count in distribution.items() if cardinality >= REVIEW_THRESHOLD)
    report = {
        "policy_version": 2,
        "input": str(args.input),
        "rows": rows,
        "invalid_rows": invalid_rows,
        "label_cardinality_2digit": {
            "mean": round(mean, 6),
            "min": min(cardinalities),
            "median": percentile(cardinalities.copy(), 0.50),
            "p95": percentile(cardinalities.copy(), 0.95),
            "p99": percentile(cardinalities.copy(), 0.99),
            "max": max(cardinalities),
            "distribution": {str(key): value for key, value in sorted(distribution.items())},
        },
        "threshold_counts_by_split": {
            split: {
                str(threshold): counts_by_split[split][threshold]
                for threshold in THRESHOLDS
            }
            for split in SPLITS
        },
        "review_threshold": REVIEW_THRESHOLD,
        "rows_at_or_above_review_threshold": reviewed_rows,
        "fraction_at_or_above_review_threshold": round(reviewed_rows / rows, 8),
        "top_archives_among_reviewed_rows": dict(high_by_archive.most_common(20)),
        "examples": high_examples[:EXAMPLE_LIMIT],
        "policy_decision": {
            "action": "retain",
            "rationale": (
                "High-cardinality records are rare, labels remain syntactically valid, "
                "and a universal cap would discard potentially legitimate multi-topic papers. "
                "Report performance by label cardinality and revisit only if training diagnostics "
                "show disproportionate influence."
            ),
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Audited {rows:,} rows")
    print(
        f"Rows with at least {REVIEW_THRESHOLD} labels: {reviewed_rows:,} "
        f"({reviewed_rows / rows:.4%})"
    )
    print(f"Wrote label-cardinality audit to {args.output}")


if __name__ == "__main__":
    main()

"""Evaluate training-only frequency thresholds for 2-digit JEL labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "repec_jel_label_vocabulary_audit.json"
SPLITS = ("train", "validation", "test", "holdout")
THRESHOLDS = (1, 10, 25, 50, 100, 250, 500, 1000)


def coverage(rows: list[list[str]], vocabulary: set[str]) -> dict[str, float | int]:
    total_instances = sum(len(labels) for labels in rows)
    retained_instances = sum(
        sum(label in vocabulary for label in labels) for labels in rows
    )
    any_label = sum(any(label in vocabulary for label in labels) for labels in rows)
    all_labels = sum(all(label in vocabulary for label in labels) for labels in rows)
    return {
        "records": len(rows),
        "records_with_any_in_vocab_label": any_label,
        "records_with_all_labels_in_vocab": all_labels,
        "label_instances": total_instances,
        "label_instances_in_vocab": retained_instances,
        "label_instance_coverage": round(retained_instances / total_instances, 6)
        if total_instances
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Clean dataset not found: {args.input}")

    rows_by_split: dict[str, list[list[str]]] = {split: [] for split in SPLITS}
    frequency_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            split = row["split"]
            labels = row["labels_2digit"]
            rows_by_split[split].append(labels)
            frequency_by_split[split].update(labels)

    train_frequency = frequency_by_split["train"]
    threshold_results = {}
    for threshold in THRESHOLDS:
        vocabulary = {label for label, count in train_frequency.items() if count >= threshold}
        threshold_results[str(threshold)] = {
            "vocabulary_size": len(vocabulary),
            "vocabulary": sorted(vocabulary),
            "coverage_by_split": {
                split: coverage(rows_by_split[split], vocabulary) for split in SPLITS
            },
        }

    report = {
        "input": str(args.input),
        "training_label_frequency": dict(sorted(train_frequency.items())),
        "label_frequency_by_split": {
            split: dict(sorted(frequency_by_split[split].items())) for split in SPLITS
        },
        "threshold_results": threshold_results,
        "labels_missing_from_training": sorted(
            set().union(*(set(frequency_by_split[split]) for split in SPLITS[1:]))
            - set(train_frequency)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote label-vocabulary audit to {args.output}")
    for threshold, result in threshold_results.items():
        test_coverage = result["coverage_by_split"]["test"]["label_instance_coverage"]
        print(
            f"minimum {threshold:>4} train examples: "
            f"{result['vocabulary_size']:>3} labels; "
            f"test label-instance coverage {test_coverage:.2%}"
        )


if __name__ == "__main__":
    main()

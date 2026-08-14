"""Create model-ready JEL dataset v3 with broad two-digit codes excluded only from that head.

The cleaned input's ``labels_1digit`` and ``labels_2digit`` fields are source
labels and are copied without modification.  Supervision fields are emitted
separately so that the 1-digit task retains every parent letter, while the
2-digit task excludes broad ``*0`` categories and labels outside the
training-only vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_model_v3.jsonl"
DEFAULT_PARQUET_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_model_v3.parquet"
DEFAULT_VOCABULARY = PROJECT_ROOT / "data" / "jel_2digit_vocabulary_v3.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "repec_jel_model_v3_report.json"
POLICY_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parquet-output", type=Path, default=DEFAULT_PARQUET_OUTPUT)
    parser.add_argument("--vocabulary-output", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-train-label-count", type=int, default=50)
    return parser.parse_args()


def is_broad_2digit_code(label: str) -> bool:
    """Return whether a normalized two-digit JEL label is its broad *0 code."""
    return len(label) == 2 and label[0].isalpha() and label[1] == "0"


def two_digit_candidates(labels: list[str]) -> list[str]:
    """Remove only broad *0 labels, retaining source order for target labels."""
    return [label for label in labels if not is_broad_2digit_code(label)]


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Clean dataset not found: {args.input}")
    if args.min_train_label_count < 1:
        raise ValueError("--min-train-label-count must be positive")
    for path in (args.output, args.parquet_output, args.vocabulary_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet output requires pyarrow. Install it with: uv sync") from error

    train_frequency = Counter()
    broad_label_instances = Counter()
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["split"] == "train":
                labels = row["labels_2digit"]
                train_frequency.update(two_digit_candidates(labels))
                broad_label_instances.update(label for label in labels if is_broad_2digit_code(label))

    vocabulary = sorted(
        label for label, count in train_frequency.items() if count >= args.min_train_label_count
    )
    vocabulary_set = set(vocabulary)
    schema = pa.schema([
        ("pid", pa.int64()),
        ("handle", pa.string()),
        ("year", pa.int64()),
        ("split", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("text", pa.string()),
        ("labels_1digit", pa.list_(pa.string())),
        ("labels_2digit", pa.list_(pa.string())),
        ("labels_1digit_target", pa.list_(pa.string())),
        ("labels_2digit_target", pa.list_(pa.string())),
        ("has_2digit_target", pa.bool_()),
    ])
    columns = [field.name for field in schema]
    raw_by_split = Counter()
    target_2digit_by_split = Counter()
    no_2digit_target_by_split = Counter()
    broad_by_split = Counter()
    out_of_vocabulary_by_split = Counter()
    jsonl_temporary = args.output.with_name(args.output.name + ".tmp")
    parquet_temporary = args.parquet_output.with_name(args.parquet_output.name + ".tmp")
    parquet_writer = pq.ParquetWriter(parquet_temporary, schema, compression="zstd")
    parquet_buffer: list[dict[str, object]] = []
    with args.input.open(encoding="utf-8") as source, jsonl_temporary.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            raw_2digit = row["labels_2digit"]
            candidates = two_digit_candidates(raw_2digit)
            targets = [label for label in candidates if label in vocabulary_set]
            split = row["split"]
            row["labels_1digit_target"] = list(row["labels_1digit"])
            row["labels_2digit_target"] = targets
            row["has_2digit_target"] = bool(targets)
            raw_by_split[split] += 1
            target_2digit_by_split[split] += len(targets)
            broad_by_split[split] += sum(is_broad_2digit_code(label) for label in raw_2digit)
            out_of_vocabulary_by_split[split] += sum(label not in vocabulary_set for label in candidates)
            if not targets:
                no_2digit_target_by_split[split] += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            parquet_buffer.append({column: row[column] for column in columns})
            if len(parquet_buffer) == 10_000:
                parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=schema))
                parquet_buffer.clear()
    if parquet_buffer:
        parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=schema))
    parquet_writer.close()
    os.replace(jsonl_temporary, args.output)
    os.replace(parquet_temporary, args.parquet_output)

    write_json_atomic(args.vocabulary_output, {
        "policy_version": POLICY_VERSION,
        "minimum_training_examples": args.min_train_label_count,
        "training_split": {"name": "train", "years": [2015, 2023]},
        "broad_2digit_codes_excluded": "*0",
        "labels_2digit": vocabulary,
        "training_frequency": {label: train_frequency[label] for label in vocabulary},
    })
    report = {
        "policy_version": POLICY_VERSION,
        "input": str(args.input),
        "output": str(args.output),
        "parquet_output": str(args.parquet_output),
        "vocabulary_output": str(args.vocabulary_output),
        "minimum_training_examples": args.min_train_label_count,
        "vocabulary_size": len(vocabulary),
        "metrics": {
            "records_by_split": dict(sorted(raw_by_split.items())),
            "two_digit_target_instances_by_split": dict(sorted(target_2digit_by_split.items())),
            "records_without_2digit_target_by_split": dict(sorted(no_2digit_target_by_split.items())),
            "broad_2digit_instances_excluded_by_split": dict(sorted(broad_by_split.items())),
            "non_broad_2digit_instances_outside_vocabulary_by_split": dict(sorted(out_of_vocabulary_by_split.items())),
            "broad_2digit_training_frequency": dict(sorted(broad_label_instances.items())),
        },
    }
    write_json_atomic(args.report, report)
    print(f"Wrote {sum(raw_by_split.values()):,} model-ready records to {args.output}")
    print(f"Wrote Parquet dataset to {args.parquet_output}")
    print(f"2-digit vocabulary: {len(vocabulary)} labels")


if __name__ == "__main__":
    main()

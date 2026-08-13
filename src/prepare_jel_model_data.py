"""Apply a training-only 2-digit JEL vocabulary to the cleaned dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_model_v1.jsonl"
DEFAULT_PARQUET_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_model_v1.parquet"
DEFAULT_VOCABULARY = PROJECT_ROOT / "data" / "jel_2digit_vocabulary_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "repec_jel_model_v1_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parquet-output", type=Path, default=DEFAULT_PARQUET_OUTPUT)
    parser.add_argument("--vocabulary-output", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-train-label-count", type=int, default=50)
    parser.add_argument("--policy-version", type=int, default=1)
    return parser.parse_args()


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
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it with: "
            "uv sync"
        ) from error

    train_frequency = Counter()
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["split"] == "train":
                train_frequency.update(row["labels_2digit"])
    vocabulary = sorted(
        label
        for label, count in train_frequency.items()
        if count >= args.min_train_label_count
    )
    vocabulary_set = set(vocabulary)

    kept_by_split = Counter()
    dropped_by_split = Counter()
    removed_label_instances = Counter()
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
    ])
    jsonl_temporary = args.output.with_name(args.output.name + ".tmp")
    parquet_temporary = args.parquet_output.with_name(args.parquet_output.name + ".tmp")
    parquet_writer = pq.ParquetWriter(parquet_temporary, schema, compression="zstd")
    parquet_buffer: list[dict[str, object]] = []
    parquet_columns = [field.name for field in schema]
    with args.input.open(encoding="utf-8") as source, jsonl_temporary.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            original_labels = row["labels_2digit"]
            retained_labels = [label for label in original_labels if label in vocabulary_set]
            removed_label_instances.update(
                label for label in original_labels if label not in vocabulary_set
            )
            if not retained_labels:
                dropped_by_split[row["split"]] += 1
                continue

            row["labels_2digit"] = retained_labels
            row["labels_1digit"] = sorted({label[0] for label in retained_labels})
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            parquet_buffer.append({column: row[column] for column in parquet_columns})
            if len(parquet_buffer) == 10_000:
                parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=schema))
                parquet_buffer.clear()
            kept_by_split[row["split"]] += 1
    if parquet_buffer:
        parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=schema))
    parquet_writer.close()
    os.replace(jsonl_temporary, args.output)
    os.replace(parquet_temporary, args.parquet_output)

    vocabulary_payload = {
        "policy_version": args.policy_version,
        "minimum_training_examples": args.min_train_label_count,
        "labels_2digit": vocabulary,
        "training_frequency": {label: train_frequency[label] for label in vocabulary},
    }
    vocabulary_temporary = args.vocabulary_output.with_name(
        args.vocabulary_output.name + ".tmp"
    )
    vocabulary_temporary.write_text(
        json.dumps(vocabulary_payload, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(vocabulary_temporary, args.vocabulary_output)
    report = {
        "policy_version": args.policy_version,
        "input": str(args.input),
        "output": str(args.output),
        "parquet_output": str(args.parquet_output),
        "vocabulary_output": str(args.vocabulary_output),
        "minimum_training_examples": args.min_train_label_count,
        "vocabulary_size": len(vocabulary),
        "kept_by_split": dict(sorted(kept_by_split.items())),
        "dropped_rows_with_no_in_vocabulary_labels_by_split": dict(sorted(dropped_by_split.items())),
        "removed_label_instances": dict(sorted(removed_label_instances.items())),
    }
    report_temporary = args.report.with_name(args.report.name + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, args.report)
    print(f"Wrote {sum(kept_by_split.values()):,} model-ready records to {args.output}")
    print(f"Wrote Parquet dataset to {args.parquet_output}")
    print(f"2-digit vocabulary: {len(vocabulary)} labels")


if __name__ == "__main__":
    main()

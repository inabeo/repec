"""Create a checksummed provenance manifest for the frozen JEL dataset v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "repec.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "repec_jel_model_v2_manifest.json"
DEFAULT_ARTIFACTS = (
    "data/repec_jel_2015_2026_raw.jsonl",
    "data/repec_jel_duplicate_map_v2.jsonl",
    "data/repec_jel_2015_2026_clean_v2.jsonl",
    "data/repec_jel_2015_2026_excluded_v2.jsonl",
    "data/repec_jel_2015_2026_model_v2.jsonl",
    "data/repec_jel_2015_2026_model_v2.parquet",
    "data/jel_2digit_vocabulary_v2.json",
    "reports/repec_jel_near_duplicate_audit_v2.json",
    "reports/repec_jel_clean_v2_report.json",
    "reports/repec_jel_label_cardinality_v2.json",
    "reports/repec_jel_label_vocabulary_audit_v2.json",
    "reports/repec_jel_model_v2_report.json",
)
PIPELINE_SCRIPTS = (
    "src/extract_jel_training_data.py",
    "src/jel_duplicate_utils.py",
    "src/audit_jel_near_duplicates.py",
    "src/clean_jel_training_data_v2.py",
    "src/audit_jel_label_cardinality.py",
    "src/audit_jel_label_vocabulary.py",
    "src/prepare_jel_model_data.py",
    "src/create_jel_dataset_manifest.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-refresh-date",
        help="Source refresh date (YYYY-MM-DD); defaults to the database mtime date in UTC",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def file_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "modified_at_utc": timestamp(path),
        "sha256": sha256(path),
    }


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    artifact_paths = [PROJECT_ROOT / relative for relative in DEFAULT_ARTIFACTS]
    script_paths = [PROJECT_ROOT / relative for relative in PIPELINE_SCRIPTS]
    missing = [str(path) for path in (*artifact_paths, *script_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest inputs missing: {missing}")

    clean_report = json.loads(
        (PROJECT_ROOT / "reports/repec_jel_clean_v2_report.json").read_text()
    )
    duplicate_report = json.loads(
        (PROJECT_ROOT / "reports/repec_jel_near_duplicate_audit_v2.json").read_text()
    )
    cardinality_report = json.loads(
        (PROJECT_ROOT / "reports/repec_jel_label_cardinality_v2.json").read_text()
    )
    model_report = json.loads(
        (PROJECT_ROOT / "reports/repec_jel_model_v2_report.json").read_text()
    )
    vocabulary = json.loads(
        (PROJECT_ROOT / "data/jel_2digit_vocabulary_v2.json").read_text()
    )

    status = git_output("status", "--porcelain")
    source_refresh_date = args.source_refresh_date or datetime.fromtimestamp(
        database.stat().st_mtime, timezone.utc
    ).date().isoformat()
    manifest = {
        "manifest_version": 1,
        "dataset_policy_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_database": {
            **file_entry(database),
            "refresh_completed_date": source_refresh_date,
            "refresh_command": "python repec/main.py update --database ./repec.db",
        },
        "dataset_definition": {
            "language": "en",
            "year_range": [2015, 2026],
            "splits": {
                "train": [2015, 2023],
                "validation": [2024, 2024],
                "test": [2025, 2025],
                "holdout": [2026, 2026],
            },
            "holdout_note": "2026 is a partial-year snapshot as of the source refresh date.",
            "minimum_training_examples_per_2digit_label": model_report[
                "minimum_training_examples"
            ],
            "label_vocabulary_size_2digit": len(vocabulary["labels_2digit"]),
            "label_vocabulary_2digit": vocabulary["labels_2digit"],
            "high_cardinality_policy": cardinality_report["policy_decision"],
            "duplicate_matching_policy": duplicate_report["matching_policy"],
        },
        "record_counts": {
            "raw": clean_report["input_rows"],
            "clean_v2": clean_report["kept_rows"],
            "model_v2_by_split": model_report["kept_by_split"],
            "model_v2_total": sum(model_report["kept_by_split"].values()),
            "excluded_v2": clean_report["excluded_rows"],
            "excluded_v2_by_reason": clean_report["excluded_by_reason"],
        },
        "environment": {
            "python": platform.python_version(),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "platform": platform.platform(),
            "uv_lock_sha256": sha256(PROJECT_ROOT / "uv.lock"),
        },
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "working_tree_clean": not bool(status),
            "working_tree_status": status.splitlines(),
        },
        "pipeline_scripts": [file_entry(path) for path in script_paths],
        "artifacts": [file_entry(path) for path in artifact_paths],
        "pipeline_commands": [
            "uv run python src/extract_jel_training_data.py",
            "uv run python src/audit_jel_near_duplicates.py",
            "uv run python src/clean_jel_training_data_v2.py",
            "uv run python src/audit_jel_label_cardinality.py",
            (
                "uv run python src/audit_jel_label_vocabulary.py "
                "--input data/repec_jel_2015_2026_clean_v2.jsonl "
                "--output reports/repec_jel_label_vocabulary_audit_v2.json"
            ),
            (
                "uv run python src/prepare_jel_model_data.py "
                "--input data/repec_jel_2015_2026_clean_v2.jsonl "
                "--output data/repec_jel_2015_2026_model_v2.jsonl "
                "--parquet-output data/repec_jel_2015_2026_model_v2.parquet "
                "--vocabulary-output data/jel_2digit_vocabulary_v2.json "
                "--report reports/repec_jel_model_v2_report.json "
                "--min-train-label-count 50 --policy-version 2"
            ),
            "uv run python src/create_jel_dataset_manifest.py",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Wrote provenance manifest to {args.output}")
    print(f"Source database SHA-256: {manifest['source_database']['sha256']}")


if __name__ == "__main__":
    main()

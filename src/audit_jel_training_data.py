"""Audit a raw JEL-training JSONL extract without modifying it.

The report is intended to set cleaning rules from observed data, rather than
hard-coding arbitrary deletion thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "repec_jel_raw_audit.json"
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def percentile(values: list[int], probability: float) -> int | None:
    """Nearest-rank percentile; adequate for transparent audit summaries."""
    if not values:
        return None
    values.sort()
    index = round((len(values) - 1) * probability)
    return values[index]


def summarise_lengths(values: list[int]) -> dict[str, int | None]:
    return {
        "min": min(values) if values else None,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def normalise_for_exact_duplicate_check(title: str, abstract: str) -> str:
    """Use only for later exact-duplicate analysis, not to alter model text."""
    return WHITESPACE.sub(" ", f"{title} {abstract}").strip().casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw extract not found: {args.input}")

    counts = Counter()
    labels_1digit = Counter()
    labels_2digit = Counter()
    labels_1digit_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    labels_2digit_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    title_lengths: list[int] = []
    abstract_lengths: list[int] = []
    text_lengths: list[int] = []
    artifact_counts = Counter()
    invalid_rows = 0

    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
                title = row["title"]
                abstract = row["abstract"]
                split = row["split"]
                one_digit = row["labels_1digit"]
                two_digit = row["labels_2digit"]
            except (json.JSONDecodeError, KeyError, TypeError):
                invalid_rows += 1
                continue

            counts["records"] += 1
            counts[f"split:{split}"] += 1
            labels_1digit.update(one_digit)
            labels_2digit.update(two_digit)
            labels_1digit_by_split[split].update(one_digit)
            labels_2digit_by_split[split].update(two_digit)
            title_lengths.append(len(title))
            abstract_lengths.append(len(abstract))
            text_lengths.append(len(row.get("text", "")))

            for field_name, value in (("title", title), ("abstract", abstract)):
                if HTML_TAG.search(value):
                    artifact_counts[f"html_tag:{field_name}"] += 1
                if "\ufffd" in value:
                    artifact_counts[f"replacement_character:{field_name}"] += 1
                if URL.search(value):
                    artifact_counts[f"url:{field_name}"] += 1
                if value != value.strip():
                    artifact_counts[f"outer_whitespace:{field_name}"] += 1
                if "\x00" in value:
                    artifact_counts[f"null_character:{field_name}"] += 1

    report = {
        "input": str(args.input),
        "records": counts["records"],
        "invalid_json_or_schema_rows": invalid_rows,
        "records_by_split": {
            split: counts[f"split:{split}"]
            for split in ("train", "validation", "test", "holdout")
        },
        "character_lengths": {
            "title": summarise_lengths(title_lengths),
            "abstract": summarise_lengths(abstract_lengths),
            "combined_text": summarise_lengths(text_lengths),
        },
        "text_artifacts": dict(sorted(artifact_counts.items())),
        "labels": {
            "one_digit_frequency": dict(sorted(labels_1digit.items())),
            "two_digit_frequency": dict(sorted(labels_2digit.items())),
            "one_digit_frequency_by_split": {
                split: dict(sorted(labels_1digit_by_split[split].items()))
                for split in ("train", "validation", "test", "holdout")
            },
            "two_digit_frequency_by_split": {
                split: dict(sorted(labels_2digit_by_split[split].items()))
                for split in ("train", "validation", "test", "holdout")
            },
        },
        "notes": [
            "This report makes no deletion or text transformation decisions.",
            "Exact and near-duplicate analysis is intentionally a separate next step.",
            "Choose all thresholds using the training split only before applying them to later splits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote audit report to {args.output}")
    print(f"Audited {counts['records']:,} records")


if __name__ == "__main__":
    main()

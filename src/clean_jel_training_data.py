"""Create a conservative cleaned JEL-training dataset from the raw JSONL.

Policy version 1:
* normalize text without changing the source extract;
* remove clear placeholder/too-short abstracts;
* retain one earliest record for exact duplicate text with identical labels;
* exclude every exact-duplicate group with conflicting two-digit labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from html import unescape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v1.jsonl"
DEFAULT_EXCLUDED = PROJECT_ROOT / "data" / "repec_jel_2015_2026_excluded_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "repec_jel_clean_v1_report.json"
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
WHITESPACE = re.compile(r"\s+")
PLACEHOLDER_ABSTRACTS = {
    "no abstract",
    "t.b.d.",
    "tbd",
    "error",
    "contents",
    "foreword",
    "editorial",
    "interviews",
}
MIN_ABSTRACT_CHARACTERS = 50


def clean_text(value: str) -> str:
    """Perform narrowly scoped normalization suitable for model input."""
    value = unescape(value).replace("\ufffd", " ")
    value = HTML_TAG.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def prepare_row(row: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    """Return normalized row or a deterministic exclusion reason."""
    title = clean_text(str(row["title"]))
    abstract = clean_text(str(row["abstract"]))
    normalised_abstract = abstract.casefold()
    if normalised_abstract in PLACEHOLDER_ABSTRACTS:
        return None, "placeholder_abstract"
    if len(abstract) < MIN_ABSTRACT_CHARACTERS:
        return None, "abstract_under_50_characters"

    prepared = dict(row)
    prepared["title"] = title
    prepared["abstract"] = abstract
    prepared["text"] = f"{title} {abstract}"
    return prepared, None


def fingerprint(row: dict[str, object]) -> str:
    """Exact-duplicate key after deterministic normalization."""
    text = f"{row['title']}\n{row['abstract']}"
    text = WHITESPACE.sub(" ", text).strip().casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--excluded-output", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def write_jsonl(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw extract not found: {args.input}")
    for path in (args.output, args.excluded_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    # First pass determines the representative of every normalized text group.
    # Only metadata is held in memory; title/abstract text remains on disk.
    groups: dict[str, dict[str, object]] = {}
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            row, reason = prepare_row(raw)
            if reason:
                continue
            assert row is not None
            key = fingerprint(row)
            label_signature = tuple(row["labels_2digit"])
            rank = (int(row["year"]), int(row["pid"]))
            group = groups.get(key)
            if group is None:
                groups[key] = {
                    "representative_rank": rank,
                    "label_signatures": {label_signature},
                }
            else:
                labels = group["label_signatures"]
                assert isinstance(labels, set)
                labels.add(label_signature)
                if rank < group["representative_rank"]:
                    group["representative_rank"] = rank

    exclusion_counts = Counter()
    kept_by_split = Counter()
    excluded_by_split = Counter()
    label_counts_train = Counter()
    input_rows = 0
    with (
        args.input.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as cleaned,
        args.excluded_output.open("w", encoding="utf-8") as excluded,
    ):
        for line in source:
            input_rows += 1
            raw = json.loads(line)
            row, reason = prepare_row(raw)
            if reason is None:
                assert row is not None
                group = groups[fingerprint(row)]
                labels = group["label_signatures"]
                assert isinstance(labels, set)
                if len(labels) > 1:
                    reason = "conflicting_labels_for_exact_duplicate_text"
                elif (int(row["year"]), int(row["pid"])) != group["representative_rank"]:
                    reason = "duplicate_text_with_identical_labels"

            if reason:
                excluded_row = dict(raw)
                excluded_row["removal_reason"] = reason
                write_jsonl(excluded, excluded_row)
                exclusion_counts[reason] += 1
                excluded_by_split[str(raw["split"])] += 1
                continue

            assert row is not None
            write_jsonl(cleaned, row)
            split = str(row["split"])
            kept_by_split[split] += 1
            if split == "train":
                label_counts_train.update(row["labels_2digit"])

    report = {
        "policy_version": 1,
        "input": str(args.input),
        "output": str(args.output),
        "excluded_output": str(args.excluded_output),
        "min_abstract_characters": MIN_ABSTRACT_CHARACTERS,
        "placeholder_abstracts": sorted(PLACEHOLDER_ABSTRACTS),
        "input_rows": input_rows,
        "kept_rows": sum(kept_by_split.values()),
        "kept_by_split": dict(sorted(kept_by_split.items())),
        "excluded_rows": sum(exclusion_counts.values()),
        "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        "excluded_by_split": dict(sorted(excluded_by_split.items())),
        "training_2digit_label_frequency": dict(sorted(label_counts_train.items())),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report['kept_rows']:,} cleaned records to {args.output}")
    print(f"Wrote {report['excluded_rows']:,} excluded records to {args.excluded_output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()

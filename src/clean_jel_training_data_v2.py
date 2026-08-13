"""Create the temporally safe, version-deduplicated JEL dataset v2."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from jel_duplicate_utils import (
    MIN_ABSTRACT_CHARACTERS,
    PLACEHOLDER_ABSTRACTS,
    prepare_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"
DEFAULT_MAP = PROJECT_ROOT / "data" / "repec_jel_duplicate_map_v2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_clean_v2.jsonl"
DEFAULT_EXCLUDED = PROJECT_ROOT / "data" / "repec_jel_2015_2026_excluded_v2.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "repec_jel_clean_v2_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--duplicate-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--excluded-output", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.input, args.duplicate_map):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    for path in (args.output, args.excluded_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    duplicate_map: dict[int, dict[str, object]] = {}
    with args.duplicate_map.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            pid = int(item["pid"])
            if pid in duplicate_map:
                raise ValueError(f"Duplicate pid in duplicate map: {pid}")
            duplicate_map[pid] = item

    exclusion_counts = Counter()
    excluded_by_split = Counter()
    kept_by_split = Counter()
    label_counts_train = Counter()
    input_rows = 0
    consumed_duplicate_pids: set[int] = set()
    output_temporary = args.output.with_name(args.output.name + ".tmp")
    excluded_temporary = args.excluded_output.with_name(args.excluded_output.name + ".tmp")
    with (
        args.input.open(encoding="utf-8") as source,
        output_temporary.open("w", encoding="utf-8") as cleaned,
        excluded_temporary.open("w", encoding="utf-8") as excluded,
    ):
        for line in source:
            input_rows += 1
            raw = json.loads(line)
            row, reason = prepare_row(raw)
            duplicate = duplicate_map.get(int(raw["pid"]))
            if reason is None and duplicate is not None:
                reason = str(duplicate["removal_reason"])
                consumed_duplicate_pids.add(int(raw["pid"]))

            if reason:
                excluded_row = dict(raw)
                excluded_row["removal_reason"] = reason
                if duplicate is not None and reason == duplicate["removal_reason"]:
                    excluded_row["duplicate_of_pid"] = duplicate["representative_pid"]
                    excluded_row["duplicate_family_id"] = duplicate["family_id"]
                excluded.write(json.dumps(excluded_row, ensure_ascii=False) + "\n")
                exclusion_counts[reason] += 1
                excluded_by_split[str(raw["split"])] += 1
                continue

            assert row is not None
            cleaned.write(json.dumps(row, ensure_ascii=False) + "\n")
            split = str(row["split"])
            kept_by_split[split] += 1
            if split == "train":
                label_counts_train.update(row["labels_2digit"])

    if consumed_duplicate_pids != duplicate_map.keys():
        missing = sorted(duplicate_map.keys() - consumed_duplicate_pids)
        raise RuntimeError(
            f"Duplicate map contains {len(missing)} pids not consumed by cleaning; "
            f"first pids: {missing[:10]}"
        )
    os.replace(output_temporary, args.output)
    os.replace(excluded_temporary, args.excluded_output)

    report = {
        "policy_version": 2,
        "input": str(args.input),
        "duplicate_map": str(args.duplicate_map),
        "output": str(args.output),
        "excluded_output": str(args.excluded_output),
        "min_abstract_characters": MIN_ABSTRACT_CHARACTERS,
        "placeholder_abstracts": sorted(PLACEHOLDER_ABSTRACTS),
        "duplicate_policy": {
            "matching_uses_labels_or_splits": False,
            "representative": "earliest (year, pid)",
            "conflicting_labels": "do not affect matching or representative selection",
        },
        "input_rows": input_rows,
        "kept_rows": sum(kept_by_split.values()),
        "kept_by_split": dict(sorted(kept_by_split.items())),
        "excluded_rows": sum(exclusion_counts.values()),
        "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        "excluded_by_split": dict(sorted(excluded_by_split.items())),
        "training_2digit_label_frequency": dict(sorted(label_counts_train.items())),
    }
    report_temporary = args.report.with_name(args.report.name + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, args.report)
    print(f"Wrote {report['kept_rows']:,} cleaned records to {args.output}")
    print(f"Wrote {report['excluded_rows']:,} excluded records to {args.excluded_output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()

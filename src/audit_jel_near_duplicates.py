"""Audit exact and conservative near-duplicate paper versions for dataset v2."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from jel_duplicate_utils import (
    EXACT_ABSTRACT_TITLE_JACCARD,
    EXACT_TITLE_ABSTRACT_JACCARD,
    FUZZY_TITLE_ABSTRACT_JACCARD,
    FUZZY_TITLE_JACCARD,
    MAX_FUZZY_BLOCK_SIZE,
    MIN_ABSTRACT_TOKENS,
    MIN_FUZZY_TITLE_TOKENS,
    Document,
    build_duplicate_families,
    normalized_abstract_hash,
    normalized_text_hash,
    normalized_words,
    prepare_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"
DEFAULT_MAP = PROJECT_ROOT / "data" / "repec_jel_duplicate_map_v2.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "repec_jel_near_duplicate_audit_v2.json"
EXAMPLE_LIMIT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--map-output", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def compact(document: Document) -> dict[str, object]:
    return {
        "pid": document.pid,
        "handle": document.handle,
        "year": document.year,
        "split": document.split,
        "title": document.title,
        "labels_2digit": list(document.labels_2digit),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw extract not found: {args.input}")
    for path in (args.map_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    text_exclusions = Counter()
    input_rows = 0
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            input_rows += 1
            raw = json.loads(line)
            row, reason = prepare_row(raw)
            if reason:
                text_exclusions[reason] += 1
                continue
            assert row is not None
            title = str(row["title"])
            abstract = str(row["abstract"])
            documents.append(
                Document(
                    pid=int(row["pid"]),
                    year=int(row["year"]),
                    split=str(row["split"]),
                    handle=str(row["handle"]),
                    title=title,
                    abstract=abstract,
                    labels_2digit=tuple(row["labels_2digit"]),
                    title_words=normalized_words(title),
                    exact_text_hash=normalized_text_hash(title, abstract),
                    exact_abstract_hash=normalized_abstract_hash(abstract),
                )
            )

    representative_by_index, diagnostics = build_duplicate_families(documents)
    families = diagnostics["duplicate_families"]
    assert isinstance(families, list)

    family_counts = Counter()
    excluded_by_split = Counter()
    transitions = Counter()
    conflicting_label_families = 0
    examples: list[dict[str, object]] = []
    map_temporary = args.map_output.with_name(args.map_output.name + ".tmp")
    with map_temporary.open("w", encoding="utf-8") as output:
        for family_id, members in enumerate(
            sorted(
                families,
                key=lambda group: min(documents[index].rank for index in group),
            ),
            start=1,
        ):
            representative_index = min(members, key=lambda index: documents[index].rank)
            representative = documents[representative_index]
            label_signatures = {document.labels_2digit for document in (documents[i] for i in members)}
            if len(label_signatures) > 1:
                conflicting_label_families += 1
            family_splits = {documents[index].split for index in members}
            family_counts["families"] += 1
            family_counts["records"] += len(members)
            family_counts["additional_records"] += len(members) - 1
            if len(family_splits) > 1:
                family_counts["cross_split_families"] += 1

            member_payloads = []
            for index in sorted(members, key=lambda item: documents[item].rank):
                if index == representative_index:
                    continue
                document = documents[index]
                exact = document.exact_text_hash == representative.exact_text_hash
                reason = (
                    "exact_duplicate_of_earlier_record"
                    if exact
                    else "near_duplicate_version_of_earlier_record"
                )
                payload = {
                    "pid": document.pid,
                    "representative_pid": representative.pid,
                    "family_id": family_id,
                    "removal_reason": reason,
                }
                output.write(json.dumps(payload) + "\n")
                excluded_by_split[document.split] += 1
                transitions[f"{representative.split}->{document.split}"] += 1
                family_counts[reason] += 1
                if len(member_payloads) < 8:
                    member_payloads.append(compact(document))

            if len(examples) < EXAMPLE_LIMIT and (
                len(family_splits) > 1
                or any(
                    documents[index].exact_text_hash != representative.exact_text_hash
                    for index in members
                )
            ):
                examples.append(
                    {
                        "family_id": family_id,
                        "representative": compact(representative),
                        "other_records": member_payloads,
                        "label_signatures": len(label_signatures),
                    }
                )
    os.replace(map_temporary, args.map_output)

    report = {
        "policy_version": 2,
        "input": str(args.input),
        "map_output": str(args.map_output),
        "input_rows": input_rows,
        "eligible_rows_after_text_rules": len(documents),
        "text_exclusions": dict(sorted(text_exclusions.items())),
        "matching_policy": {
            "uses_labels_or_splits_to_create_matches": False,
            "representative": "earliest (year, pid)",
            "minimum_abstract_tokens": MIN_ABSTRACT_TOKENS,
            "exact_title_minimum_abstract_jaccard": EXACT_TITLE_ABSTRACT_JACCARD,
            "fuzzy_title_minimum_title_jaccard": FUZZY_TITLE_JACCARD,
            "fuzzy_title_minimum_abstract_jaccard": FUZZY_TITLE_ABSTRACT_JACCARD,
            "exact_abstract_minimum_title_jaccard": EXACT_ABSTRACT_TITLE_JACCARD,
            "minimum_fuzzy_title_tokens": MIN_FUZZY_TITLE_TOKENS,
            "maximum_fuzzy_block_size": MAX_FUZZY_BLOCK_SIZE,
        },
        "family_counts": dict(sorted(family_counts.items())),
        "excluded_by_split": dict(sorted(excluded_by_split.items())),
        "representative_to_excluded_split_transitions": dict(sorted(transitions.items())),
        "families_with_conflicting_2digit_label_signatures": conflicting_label_families,
        "candidate_comparisons": diagnostics["comparisons"],
        "skipped_oversized_blocks": diagnostics["skipped_oversized_blocks"],
        "tokenized_candidate_records": diagnostics["tokenized_candidate_records"],
        "examples": examples,
    }
    report_temporary = args.report.with_name(args.report.name + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, args.report)
    print(f"Audited {len(documents):,} eligible records")
    print(
        f"Found {family_counts['families']:,} duplicate/version families; "
        f"mapped {family_counts['additional_records']:,} later records"
    )
    print(f"Wrote duplicate map to {args.map_output}")
    print(f"Wrote audit report to {args.report}")


if __name__ == "__main__":
    main()

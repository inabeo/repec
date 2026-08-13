"""Create a raw, paper-level RePEc dataset for hierarchical JEL modelling.

This is deliberately an extraction step, not a cleaning step.  It leaves the
source SQLite database untouched and emits one JSON object per paper, with
separate title/abstract fields and both JEL label levels.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "repec.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "repec_jel_2015_2026_raw.jsonl"


QUERY = """
WITH eligible_papers AS (
    SELECT pid, handle, year, title, abstract
    FROM papers
    WHERE language = 'en'
      AND year BETWEEN :start_year AND :end_year
      AND title IS NOT NULL AND TRIM(title) <> ''
      AND abstract IS NOT NULL AND TRIM(abstract) <> ''
),
paper_labels AS (
    SELECT
        pj.pid,
        json_group_array(DISTINCT SUBSTR(pj.code, 1, 1)) AS labels_1digit,
        json_group_array(DISTINCT SUBSTR(pj.code, 1, 2)) AS labels_2digit
    FROM papers_jel AS pj
    JOIN eligible_papers AS p ON p.pid = pj.pid
    WHERE pj.code IS NOT NULL
      AND LENGTH(pj.code) >= 2
    GROUP BY pj.pid
)
SELECT
    p.pid,
    p.handle,
    p.year,
    p.title,
    p.abstract,
    l.labels_1digit,
    l.labels_2digit
FROM eligible_papers AS p
JOIN paper_labels AS l ON l.pid = p.pid
ORDER BY p.year, p.pid
"""


def split_for_year(year: int) -> str:
    """Return the pre-registered temporal split for a record year."""
    if 2015 <= year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    if year == 2025:
        return "test"
    if year == 2026:
        return "holdout"
    raise ValueError(f"Year {year} is outside the supported split range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database.is_file():
        raise FileNotFoundError(f"Database not found: {args.database}")
    if args.start_year < 2015 or args.end_year > 2026:
        raise ValueError("This split definition supports years from 2015 through 2026")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    # Read-only mode guarantees the raw source database is never modified.
    database_uri = f"file:{args.database.resolve()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        cursor = connection.execute(
            QUERY, {"start_year": args.start_year, "end_year": args.end_year}
        )
        columns = [column[0] for column in cursor.description]
        with args.output.open("w", encoding="utf-8") as output:
            for values in cursor:
                row = dict(zip(columns, values, strict=True))
                row["labels_1digit"] = sorted(json.loads(row.pop("labels_1digit")))
                row["labels_2digit"] = sorted(json.loads(row.pop("labels_2digit")))
                row["split"] = split_for_year(row["year"])
                row["text"] = f"{row['title']} {row['abstract']}"
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["split"]] += 1

    print(f"Wrote {sum(counts.values()):,} paper-level records to {args.output}")
    for split in ("train", "validation", "test", "holdout"):
        print(f"{split}: {counts[split]:,}")


if __name__ == "__main__":
    main()

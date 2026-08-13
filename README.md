# RePEc Database Manager

## Introduction

A collection of Python scripts to download, cleanup, and somewhat structure the [RePEc dataset](http://repec.org/). The data is downloaded into an SQLite database, so it is possible to use SQL queries to analyse the data. E.g.,

```sql
SELECT count(*) FROM papers JOIN papers_jel USING (pid) WHERE year = 2010  AND code = 'D43'
```

will show how many papers were written in 2010 about oligopolistic markets.

## Getting Started
Run

```bash
python main.py init
python main.py update
```

to setup an empty SQLite database, `repec.db`, and to download the full RePEc dataset into it (takes a while). See

```bash
python main.py init --help
python main.py update --help
```

for available options.

## Non-standard Dependencies

The scripts use [cld2-cffi](https://github.com/GregBowyer/cld2-cffi) for automatic language detection, and [curl](https://curl.se/) for downloading from FTP sites. `curl` is used instead of `requests`, because `requests` cannot handle some of the FTP sites out there.

## Update Process

The script downloads the data using breadth-first approach. First, the names of all the available ReDIF files are downloaded from the RePEc FTP and saved in table `repec`. Second, all the files from table `repec` are downloaded from the RePEc FTP and are used to fill in table `series`. Among other data, table `series` will contain URLs where the data on particular series can be found. Unique URLs are then saved in table `remotes`. Third, all unique URLs will be visited to collect the listings of the final ReDIF documents. These listings are saved in table `listings`. Fourth, all the files from table `listings` are downloaded, processed, and saved in tables `papers`, `authors`, and `papers_jel`.

If an update is interrupted during the last stage, you can run
```bash
python main.py update --papers
```

and the update should resume from where it has stopped.

Incremental updates are currently not supported, however it is possible to perform a full update on an existing database. Paper records that are obsolete, i.e. those that can no longer be reached from the initial list of series from the RePEc FTP, are not pruned. This is done on purpose as on some days some participating websites work, and on other days they don't.

Downloaded records are saved as is in `papers.redif` (z-compressed). Additionally, the records are cleaned up and partially destructured into the respective fields. The cleanup steps include, among other:

- stripping html tags;
- language auto-detection (using [cld2-cffi](https://github.com/GregBowyer/cld2-cffi));
- jel codes extraction.

## Database

The SQLite database will contain the following tables.

Table      | Description
-----------|------------
repec      | A list of ReDIF files from RePEc FTP.
series     | Content of ReDIF files from RePEc FTP.
remotes    | A list of URLs that host RePEc data.
listings   | File listings from the sites in `remotes`.
papers     | Titles, abstracts, etc. of economic papers.
authors    | Author names.
jel        | JEL codes.
papers_jel | Correspondence between `papers` and `jel`.

## JEL Training Dataset Pipeline

This repository also contains a reproducible pipeline for creating a
paper-level dataset for hierarchical JEL multi-label classification. It uses
English titles and abstracts, predicts both 1-digit and 2-digit JEL labels,
and keeps the raw database and raw extract unchanged.

### 1. Create or refresh the source database

From the repository root, install the pinned project environment with
[uv](https://docs.astral.sh/uv/). uv creates and manages `.venv` using the
locked dependency set in `uv.lock`. This project pins Python 3.13 in
`.python-version`.

```bash
uv sync
```

This installs only the dependencies required to run the extraction, audit,
cleaning, and Parquet preparation pipeline. The historical downloader's
dependencies are isolated in the optional `downloader` extra.

### Legacy database refresh compatibility

The original RePEc downloader uses `cld2-cffi` for language detection. That
native extension currently fails to build with the current macOS compiler, so
it is deliberately excluded from the reproducible uv environment. The data
preparation and training pipeline below do not require it.

For a new database or a later full refresh, use a previously working downloader
environment until that legacy extension is replaced or patched:

```bash
python repec/main.py init --database ./repec.db
python repec/main.py update --database ./repec.db
```

For a full refresh of an existing database, run only the second command. A
full refresh can take many hours. If the paper-download stage is interrupted,
resume it with:

```bash
uv run python repec/main.py update --database ./repec.db --papers
```

### 2. Extract the raw paper-level dataset

The extractor produces one JSON object per paper. It retains separate
`title` and `abstract` fields, combined `text`, 1-digit and 2-digit label
arrays, and a temporal split:

| Split | Years | Purpose |
|---|---|---|
| `train` | 2015–2023 | model fitting |
| `validation` | 2024 | hyperparameters and thresholds |
| `test` | 2025 | locked final evaluation |
| `holdout` | 2026 | later/live evaluation |

```bash
uv run python src/extract_jel_training_data.py
uv run python src/audit_jel_training_data.py
```

The raw extract is written to `data/repec_jel_2015_2026_raw.jsonl`. Do not
modify it; it is the audit-friendly input to all later steps.

### 3. Audit and clean conservatively

Run the exact-duplicate audit before cleaning:

```bash
uv run python src/audit_jel_exact_duplicates.py
```

Then create the versioned cleaned dataset:

```bash
uv run python src/clean_jel_training_data.py
```

Cleaning policy v1 normalizes whitespace and residual HTML, replaces encoding
replacement characters, removes clear placeholder/very short abstracts, and
handles exact duplicate text. It preserves one earliest record for an
identical-label duplicate group and excludes duplicate groups whose 2-digit
labels conflict. All excluded records are retained with an explicit
`removal_reason` in `data/repec_jel_2015_2026_excluded_v1.jsonl`.

### 4. Fix the model label vocabulary and write Parquet

First inspect label support using the training period only:

```bash
uv run python src/audit_jel_label_vocabulary.py
```

Then produce the model-ready artifacts:

```bash
uv run python src/prepare_jel_model_data.py
```

The current v1 policy retains 2-digit labels with at least 50 occurrences in
the 2015–2023 training split. This yields 134 labels. The script writes:

| File | Purpose |
|---|---|
| `data/repec_jel_2015_2026_model_v1.parquet` | model input; 1 row per paper |
| `data/repec_jel_2015_2026_model_v1.jsonl` | readable/auditable equivalent |
| `data/jel_2digit_vocabulary_v1.json` | fixed 2-digit label ordering and training frequencies |
| `reports/repec_jel_model_v1_report.json` | retained/dropped record summary |

The Parquet file contains `pid`, `handle`, `year`, `split`, `title`,
`abstract`, `text`, `labels_1digit`, and `labels_2digit`. Both label columns
are native string arrays. Use the vocabulary JSON—not incidental alphabetical
ordering in a training library—as the authoritative 2-digit head order.

### Re-running the pipeline

Run the steps in order whenever the source database is refreshed. The scripts
are deterministic for a fixed database, and their versioned outputs make it
possible to compare later cleaning or vocabulary policies without overwriting
the raw source data.

## Applications

* The other day, I made a web page where you can check trends in economics. It's like a toy version of google trends but then based on words from titles and abstracts from RePEc. Some trends are suggestive, e.g. [it's all about new results](https://dubovik.eu/blog/repec?t=replicate&t=reproduce&t=verify&t=novel).

## See Also

There is also an official Perl script for downloading the data, see [remi](https://ideas.repec.org/c/rpc/script/remi.html). Remi is aimed at downloading ReDIF files, whereas the current set of scripts is aimed at downloading and partially processing the files, with the idea of using an SQLite backend to track progress and to store the final results.

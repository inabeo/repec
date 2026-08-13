"""Shared text normalization and duplicate-family detection for dataset v2."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape


HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
WHITESPACE = re.compile(r"\s+")
WORD = re.compile(r"[a-z0-9]+")
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

# These deliberately strict thresholds favor precision over recall. A false
# match would remove a genuinely new paper, while a missed match merely leaves
# a small amount of version overlap for later robustness analysis.
EXACT_TITLE_ABSTRACT_JACCARD = 0.80
FUZZY_TITLE_JACCARD = 0.85
FUZZY_TITLE_ABSTRACT_JACCARD = 0.80
EXACT_ABSTRACT_TITLE_JACCARD = 0.80
MIN_ABSTRACT_TOKENS = 20
MIN_FUZZY_TITLE_TOKENS = 6
MAX_FUZZY_BLOCK_SIZE = 500


def clean_text(value: str) -> str:
    value = unescape(value).replace("\ufffd", " ")
    value = HTML_TAG.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def prepare_row(row: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    """Normalize model text and apply only deterministic text-quality rules."""
    title = clean_text(str(row["title"]))
    abstract = clean_text(str(row["abstract"]))
    if abstract.casefold() in PLACEHOLDER_ABSTRACTS:
        return None, "placeholder_abstract"
    if len(abstract) < MIN_ABSTRACT_CHARACTERS:
        return None, "abstract_under_50_characters"

    prepared = dict(row)
    prepared["title"] = title
    prepared["abstract"] = abstract
    prepared["text"] = f"{title} {abstract}"
    return prepared, None


def normalized_words(value: str) -> tuple[str, ...]:
    return tuple(WORD.findall(value.casefold()))


def normalized_text_hash(title: str, abstract: str) -> bytes:
    text = WHITESPACE.sub(" ", f"{title}\n{abstract}").strip().casefold()
    return hashlib.sha256(text.encode("utf-8")).digest()


def normalized_abstract_hash(abstract: str) -> bytes:
    text = WHITESPACE.sub(" ", abstract).strip().casefold()
    return hashlib.sha256(text.encode("utf-8")).digest()


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0


@dataclass(slots=True)
class Document:
    pid: int
    year: int
    split: str
    handle: str
    title: str
    abstract: str
    labels_2digit: tuple[str, ...]
    title_words: tuple[str, ...]
    exact_text_hash: bytes
    exact_abstract_hash: bytes

    @property
    def rank(self) -> tuple[int, int]:
        return self.year, self.pid


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_duplicate_families(
    documents: list[Document],
) -> tuple[dict[int, int], dict[str, object]]:
    """Return index-to-representative mapping using text only.

    Labels and split names are never used to create a match. The representative
    of each connected family is the earliest ``(year, pid)`` record.
    """
    exact_text_groups: dict[bytes, list[int]] = defaultdict(list)
    exact_title_groups: dict[str, list[int]] = defaultdict(list)
    fuzzy_title_blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)
    exact_abstract_blocks: dict[tuple[bytes, str, str], list[int]] = defaultdict(list)

    for index, document in enumerate(documents):
        exact_text_groups[document.exact_text_hash].append(index)
        title_key = " ".join(document.title_words)
        exact_title_groups[title_key].append(index)
        if len(document.title_words) >= MIN_FUZZY_TITLE_TOKENS:
            boundary = (*document.title_words[:2], *document.title_words[-2:])
            fuzzy_title_blocks[boundary].append(index)
            exact_abstract_blocks[
                (document.exact_abstract_hash, document.title_words[0], document.title_words[-1])
            ].append(index)

    union_find = UnionFind(len(documents))
    match_pairs: dict[tuple[int, int], dict[str, object]] = {}
    comparisons = Counter()
    skipped_fuzzy_blocks = Counter()

    def record_match(
        left: int,
        right: int,
        match_type: str,
        title_similarity: float,
        abstract_similarity: float,
    ) -> None:
        pair = (left, right) if left < right else (right, left)
        existing = match_pairs.get(pair)
        evidence = {
            "match_type": match_type,
            "title_jaccard": round(title_similarity, 6),
            "abstract_jaccard": round(abstract_similarity, 6),
        }
        if existing is None or (
            title_similarity + abstract_similarity
            > float(existing["title_jaccard"]) + float(existing["abstract_jaccard"])
        ):
            match_pairs[pair] = evidence
        union_find.union(left, right)

    # Exact normalized title + abstract is always the same document version.
    for group in exact_text_groups.values():
        if len(group) < 2:
            continue
        representative = min(group, key=lambda index: documents[index].rank)
        for index in group:
            if index != representative:
                record_match(representative, index, "exact_text", 1.0, 1.0)

    token_cache: dict[int, tuple[frozenset[str], frozenset[str]]] = {}

    def token_sets(index: int) -> tuple[frozenset[str], frozenset[str]]:
        cached = token_cache.get(index)
        if cached is None:
            document = documents[index]
            cached = (
                frozenset(document.title_words),
                frozenset(normalized_words(document.abstract)),
            )
            token_cache[index] = cached
        return cached

    def compare_group(group: list[int], rule: str) -> None:
        for offset, left in enumerate(group):
            left_title, left_abstract = token_sets(left)
            if len(left_abstract) < MIN_ABSTRACT_TOKENS:
                continue
            for right in group[offset + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in match_pairs:
                    continue
                right_title, right_abstract = token_sets(right)
                if len(right_abstract) < MIN_ABSTRACT_TOKENS:
                    continue
                comparisons[rule] += 1
                abstract_similarity = jaccard(left_abstract, right_abstract)
                if rule == "exact_title":
                    if abstract_similarity >= EXACT_TITLE_ABSTRACT_JACCARD:
                        record_match(left, right, rule, 1.0, abstract_similarity)
                    continue

                title_similarity = jaccard(left_title, right_title)
                if rule == "fuzzy_title":
                    if (
                        title_similarity >= FUZZY_TITLE_JACCARD
                        and abstract_similarity >= FUZZY_TITLE_ABSTRACT_JACCARD
                    ):
                        record_match(
                            left, right, rule, title_similarity, abstract_similarity
                        )
                elif rule == "exact_abstract" and title_similarity >= EXACT_ABSTRACT_TITLE_JACCARD:
                    record_match(left, right, rule, title_similarity, 1.0)

    for group in exact_title_groups.values():
        if len(group) > 1:
            compare_group(group, "exact_title")

    for group in fuzzy_title_blocks.values():
        if len(group) < 2:
            continue
        if len(group) > MAX_FUZZY_BLOCK_SIZE:
            skipped_fuzzy_blocks["fuzzy_title"] += 1
            continue
        compare_group(group, "fuzzy_title")

    for group in exact_abstract_blocks.values():
        if len(group) < 2:
            continue
        if len(group) > MAX_FUZZY_BLOCK_SIZE:
            skipped_fuzzy_blocks["exact_abstract"] += 1
            continue
        compare_group(group, "exact_abstract")

    families: dict[int, list[int]] = defaultdict(list)
    for index in range(len(documents)):
        families[union_find.find(index)].append(index)

    representative_by_index: dict[int, int] = {}
    duplicate_families: list[list[int]] = []
    for members in families.values():
        if len(members) < 2:
            continue
        representative = min(members, key=lambda index: documents[index].rank)
        duplicate_families.append(members)
        for index in members:
            representative_by_index[index] = representative

    diagnostics = {
        "match_pairs": match_pairs,
        "duplicate_families": duplicate_families,
        "comparisons": dict(sorted(comparisons.items())),
        "skipped_oversized_blocks": dict(sorted(skipped_fuzzy_blocks.items())),
        "tokenized_candidate_records": len(token_cache),
    }
    return representative_by_index, diagnostics

#!/usr/bin/env python3.13
"""Naive reference for Speed Demon 2026, Track B (Index).

    python3.13 reference/b_index.py [--data /data] [--out /out]

Reads docs.txt and queries.txt from the data directory and writes results.txt
to the output directory. The rules are in specs/b-index.md.

This is the obvious implementation, kept slow on purpose: a Python dict from
every term to the set of documents containing it, boolean operators applied
left to right with Python set operations, and phrases checked by
re-tokenizing every candidate document. It is single-threaded and keeps no
positions. Its output is the correctness gate, so it is exact, not fast.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

DOCS_FILE = "docs.txt"
QUERIES_FILE = "queries.txt"
RESULTS_FILE = "results.txt"

# A token is a maximal run of ASCII [a-z0-9] after ASCII-lowercasing the bytes.
# bytes.lower() only touches A-Z, so every byte >= 0x80 stays a separator.
TOKEN_RE = re.compile(rb"[a-z0-9]+")
EMPTY: frozenset[int] = frozenset()


def tokenize(text: bytes) -> list[bytes]:
    return TOKEN_RE.findall(text.lower())


def load_corpus(path: str) -> tuple[list[bytes], dict[bytes, set[int]]]:
    """Read docs.txt. Return the raw texts and the term -> doc id set index."""
    texts: list[bytes] = []
    index: dict[bytes, set[int]] = {}
    with open(path, "rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                line = line[:-1]
            id_field, _, text = line.partition(b"\t")
            doc_id = int(id_field)
            if doc_id != len(texts):
                raise ValueError(f"document ids must be 0..N-1 in order, got {doc_id}")
            texts.append(text)
            for term in set(tokenize(text)):
                postings = index.get(term)
                if postings is None:
                    index[term] = {doc_id}
                else:
                    postings.add(doc_id)
    return texts, index


def run_bool(index: dict[bytes, set[int]], words: list[bytes]):
    """words is t1 OP t2 OP t3 ...; apply each operator to the running result."""
    result = index.get(words[0], EMPTY)
    for op, term in zip(words[1::2], words[2::2]):
        postings = index.get(term, EMPTY)
        if op == b"AND":
            result = result & postings
        elif op == b"OR":
            result = result | postings
        elif op == b"NOT":
            result = result - postings
        else:
            raise ValueError(f"unknown operator {op!r}")
    return result


def run_phrase(index: dict[bytes, set[int]], texts: list[bytes], terms: list[bytes]) -> list[int]:
    """Documents containing every term, then an exact scan of each one."""
    candidates = index.get(terms[0], EMPTY)
    for term in terms[1:]:
        candidates = candidates & index.get(term, EMPTY)
    k = len(terms)
    matches = []
    for doc_id in candidates:
        tokens = tokenize(texts[doc_id])
        for i in range(len(tokens) - k + 1):
            if tokens[i:i + k] == terms:
                matches.append(doc_id)
                break
    return matches


def format_result(doc_ids) -> str:
    """count, sum of ids, then the (up to) 10 smallest ids ascending."""
    ids = sorted(doc_ids)
    return " ".join(str(x) for x in [len(ids), sum(ids)] + ids[:10])


def answer(index, texts, line: bytes) -> str:
    kind, _, rest = line.partition(b" ")
    if kind == b"BOOL":
        return format_result(run_bool(index, rest.split(b" ")))
    if kind == b"PHRASE":
        if not (rest.startswith(b'"') and rest.endswith(b'"')):
            raise ValueError(f"bad phrase query {line!r}")
        return format_result(run_phrase(index, texts, rest[1:-1].split(b" ")))
    raise ValueError(f"bad query {line!r}")


def main(data_dir: str, out_dir: str) -> None:
    texts, index = load_corpus(os.path.join(data_dir, DOCS_FILE))
    results = []
    with open(os.path.join(data_dir, QUERIES_FILE), "rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                line = line[:-1]
            results.append(answer(index, texts, line))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, RESULTS_FILE), "w", encoding="ascii", newline="\n") as f:
        f.write("".join(r + "\n" for r in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B (Index) naive reference.")
    parser.add_argument("--data", default="/data")
    parser.add_argument("--out", default="/out")
    args = parser.parse_args()
    main(args.data, args.out)
    sys.exit(0)

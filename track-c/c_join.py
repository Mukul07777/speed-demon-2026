#!/usr/bin/env python3.13
"""Speed Demon 2026, Track C (Join): naive reference implementation.

    python3.13 reference/c_join.py [--data /data] [--out /out]

Reads records.csv and regions.csv from the data directory and writes
aggregates.txt to the output directory, as specified in specs/c-join.md.

Deliberately the obvious version: Python's csv module does the parsing, a dict
does the join, one small object per country does the aggregation, and the
countries are sorted at the end. Correct, single-threaded, and slow.
"""

import argparse
import csv
import os
import re

RECORDS_FILE = "records.csv"
REGIONS_FILE = "regions.csv"
OUTPUT_FILE = "aggregates.txt"

RECORDS_HEADER = ["id", "name", "note", "region_id", "amount"]
REGIONS_HEADER = ["region_id", "country"]
BOM = "﻿"
UNMATCHED = "_UNMATCHED"
DIGITS = re.compile(r"[0-9]+")                     # ASCII only; int() alone would
AMOUNT = re.compile(r"(-?)([0-9]+)\.([0-9]{2})")   # also take "+5", " 5", "1_0"


def parse_int(text):
    """'42' -> 42, for id and region_id."""
    if DIGITS.fullmatch(text) is None:
        raise ValueError(f"bad integer {text!r}")
    return int(text)


def parse_cents(text):
    """'-5.25' -> -525. Exact integer cents, never a float."""
    match = AMOUNT.fullmatch(text)
    if match is None:
        raise ValueError(f"bad amount {text!r}")
    sign, whole, frac = match.groups()
    cents = int(whole) * 100 + int(frac)
    return -cents if sign else cents


def format_cents(cents):
    """-525 -> '-5.25', -5 -> '-0.05', 0 -> '0.00'."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


class Country:
    """Running aggregates for one output line."""

    def __init__(self):
        self.count = 0
        self.id_sum = 0
        self.amount_sum = 0
        self.amount_min = None
        self.amount_max = None
        self.names = set()
        self.name_bytes = 0
        self.note_bytes = 0

    def add(self, record_id, name, note, cents):
        self.count += 1
        self.id_sum += record_id
        self.amount_sum += cents
        if self.amount_min is None or cents < self.amount_min:
            self.amount_min = cents
        if self.amount_max is None or cents > self.amount_max:
            self.amount_max = cents
        self.names.add(name)
        self.name_bytes += len(name.encode("utf-8"))
        self.note_bytes += len(note.encode("utf-8"))

    def line(self, key):
        return " ".join([
            key,
            str(self.count),
            str(self.id_sum),
            format_cents(self.amount_sum),
            format_cents(self.amount_min),
            format_cents(self.amount_max),
            str(len(self.names)),
            str(self.name_bytes),
            str(self.note_bytes),
        ])


def load_regions(path):
    """regions.csv -> {region_id: country}."""
    regions = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if next(reader) != REGIONS_HEADER:
            raise ValueError("unexpected regions.csv header")
        for region_id, country in reader:
            regions[parse_int(region_id)] = country
    return regions


def read_records(path):
    """Yield (id, name, note, region_id, amount) string tuples from records.csv.

    newline="" hands CR and LF to the csv module untouched, so it can tell a
    record terminator (CRLF or LF) from an LF inside a quoted field. The excel
    dialect is the spec's quoting rule: comma separator, double-quote quoting,
    "" for a literal quote.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for number, row in enumerate(reader):
            if len(row) != 5:
                raise ValueError(f"record {number}: expected 5 fields, got {len(row)}")
            # A U+FEFF at the start of a record (the start of the file or right
            # after a record terminator) is a byte order mark, not data. The
            # first field is never quoted, so it is the first character here.
            row[0] = row[0].removeprefix(BOM)
            if number == 0:
                if row != RECORDS_HEADER:
                    raise ValueError("unexpected records.csv header")
                continue
            yield row


def main(data_dir, out_dir):
    regions = load_regions(os.path.join(data_dir, REGIONS_FILE))
    countries = {}
    for record_id, name, note, region_id, amount in read_records(os.path.join(data_dir, RECORDS_FILE)):
        key = regions.get(parse_int(region_id), UNMATCHED)    # the hash join
        country = countries.get(key)
        if country is None:
            country = countries[key] = Country()
        country.add(parse_int(record_id), name, note, parse_cents(amount))

    # str order is code point order, which is UTF-8 byte order.
    lines = [countries[key].line(key) for key in sorted(countries)]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, OUTPUT_FILE), "wb") as f:
        f.write("".join(line + "\n" for line in lines).encode("utf-8"))


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out")
    args = ap.parse_args(argv)
    main(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

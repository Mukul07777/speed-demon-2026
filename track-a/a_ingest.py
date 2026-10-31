#!/usr/bin/env python3
"""Speed Demon 2026, Track A (Ingest): the naive reference implementation.

    python3.13 reference/a_ingest.py [--data /data] [--out /out]

Reads requests.log and writes summary.txt: for every (UTC hour, endpoint) pair that
has at least one request, the request count, the p50/p95/p99 latency (nearest rank) and
the number and rate of 5xx responses. specs/a-ingest.md is the contract.

This is the obvious version on purpose, and correct: read lines, split them, parse each
timestamp with the datetime library, append each latency to a list per group, sort
every list at the end.
"""
import argparse
import os
import sys
from datetime import datetime

INPUT_NAME = "requests.log"
OUTPUT_NAME = "summary.txt"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"   # RFC 3339; %z reads the trailing Z as UTC


def percentile(sorted_values, p):
    """Nearest rank: the value at 1-indexed rank ceil(p * n / 100)."""
    n = len(sorted_values)
    rank = (p * n + 99) // 100   # ceil(p * n / 100) in integer arithmetic
    return sorted_values[rank - 1]


def main(data_dir, out_dir):
    latencies = {}   # (hour, endpoint) -> list of latency_us, in input order
    errors = {}      # (hour, endpoint) -> number of 5xx responses

    with open(os.path.join(data_dir, INPUT_NAME), encoding="utf-8") as f:
        for line in f:
            ts, method, endpoint, status, latency_us = line.rstrip("\n").split(" ")
            when = datetime.strptime(ts, TIMESTAMP_FORMAT)
            hour = when.replace(minute=0, second=0, microsecond=0)
            key = (hour, endpoint)
            if key not in latencies:
                latencies[key] = []
                errors[key] = 0
            latencies[key].append(int(latency_us))
            if 500 <= int(status) <= 599:
                errors[key] += 1

    lines = []
    # Endpoints are ASCII (see the spec), so Python's string order is bytewise order.
    for key in sorted(latencies):
        hour, endpoint = key
        values = sorted(latencies[key])
        count = len(values)
        errs = errors[key]
        ppm = errs * 1_000_000 // count
        lines.append(
            f"{hour.strftime('%Y-%m-%dT%H:00:00Z')} {endpoint} {count} "
            f"{percentile(values, 50)} {percentile(values, 95)} {percentile(values, 99)} "
            f"{errs} {ppm}\n"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, OUTPUT_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A (Ingest) reference implementation.")
    parser.add_argument("--data", default="/data", help="directory holding requests.log")
    parser.add_argument("--out", default="/out", help="directory to write summary.txt into")
    args = parser.parse_args()
    main(args.data, args.out)
    sys.exit(0)

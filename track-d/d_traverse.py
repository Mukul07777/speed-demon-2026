#!/usr/bin/env python3
"""Speed Demon 2026, Track D (Traverse): the naive reference.

    python3.13 reference/d_traverse.py [--data /data] [--out /out]

Reads graph.txt and queries.txt from the data directory and writes distances.txt
to the output directory: for each query, in input order, the length of the
shortest directed path from s to t, or -1 if t cannot be reached from s.
specs/d-traverse.md is the contract; this file is one correct way to meet it.

The approach is the textbook one: load the edges into an adjacency list of
(head, weight) pairs, then run a fresh Dijkstra search with a binary heap for
every query, stopping as soon as the target is settled.
"""

import argparse
import heapq
import os


def read_graph(path):
    """Adjacency list: adj[u] is a list of (v, w) for every edge line "u v w"."""
    with open(path, encoding="ascii") as f:
        n, m = map(int, f.readline().split())
        adj = [[] for _ in range(n)]
        for line in f:
            u, v, w = line.split()
            adj[int(u)].append((int(v), int(w)))
    return adj


def read_queries(path):
    with open(path, encoding="ascii") as f:
        return [tuple(map(int, line.split())) for line in f]


def shortest_path(adj, s, t):
    """Dijkstra from s, stopping when t is settled; -1 if t is never reached.

    Parallel edges need no special handling: each one is relaxed, and only the
    lightest can improve a distance. Weight-0 edges are fine because Dijkstra only
    needs weights to be non-negative.
    """
    dist = {s: 0}
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale entry: u was already settled at a shorter distance
        if u == t:
            return d
        for v, w in adj[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return -1


def main(data_dir, out_dir):
    adj = read_graph(os.path.join(data_dir, "graph.txt"))
    queries = read_queries(os.path.join(data_dir, "queries.txt"))
    answers = [shortest_path(adj, s, t) for s, t in queries]
    with open(os.path.join(out_dir, "distances.txt"), "w", encoding="ascii", newline="\n") as f:
        f.write("".join(f"{d}\n" for d in answers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track D (Traverse) naive reference")
    parser.add_argument("--data", default="/data")
    parser.add_argument("--out", default="/out")
    args = parser.parse_args()
    main(args.data, args.out)

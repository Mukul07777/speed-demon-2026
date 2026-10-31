# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Rust rewrite designed around the random node ids, which make every search step a cache miss. Parse into shards by node, grow compact 1024-node cells and renumber in cell order, keep edges at 4 bytes (21-bit head, 11-bit weight, about 3,500 heavy edges in a side list), and run one bidirectional Dijkstra per query on 12 threads. Laptop: 190 ms, 139 MiB | run-5754b4b1e183 | 0.1667 | 147.1 | 64.84 |
| 2 | Profile: building took 124 ms of 190, not the search as expected: growing cells 46, rewriting edges 28, reverse edges 20. Staged prefetching from the queue and from small batches cut cells and rewriting nearly in half. Reverse edges now come straight from atomic counters over the renumbered edges, with no rows. The old graph is freed when done, which it was not before. Then the search: a bucket queue replaced the binary heap and did nothing measurable (64 ms before and after), so I counted instead of guessing: 7.1M nodes settled at 72 ns each, and 532 MB of distance blocks reset. Blocks of 64 nodes instead of 1024 cut memory by 10 MiB and query work 18%. Prefetching edges on each new label cut another 8%. A branch-free label update did not help and was dropped. Laptop: 135 ms, 85 MiB | | | | |

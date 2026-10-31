# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | 893 | 0 |
| 1 | Rust rewrite designed around the random node ids, which make every search step a cache miss. Parse into shards by node, grow compact 1024-node cells and renumber in cell order, keep edges at 4 bytes (21-bit head, 11-bit weight, about 3,500 heavy edges in a side list), and run one bidirectional Dijkstra per query on 12 threads. Laptop: 190 ms, 139 MiB | | | | |

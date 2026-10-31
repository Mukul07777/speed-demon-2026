# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 14.44 | 367 | 0 |
| 1 | Rust rewrite designed from data statistics, not from profiling a naive port. 23% of records hold a `""` escape and record length varies, so fields are read with no data-dependent branch: AVX2 structure masks, a carry-less multiply for the inside-quotes mask, and quote counts from rank queries. Names are uniform over 100,000 values, so every lookup misses cache: a batch prefetches slots, then entries, and an entry holds the name and its country bits in one cache line. Laptop: 36 ms, 12 MiB. Checked against the seed-1 hash and hostile cases compared with an independent `csv` module answer | run-b435db3de7a2 | 0.0514 | 13.2 | 70.62 |

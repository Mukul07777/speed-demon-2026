# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | Rust rewrite: parallel block reads, u64 rows bucketed by hour, one full sort per hour. Profile on this version: parsing 44 ms of 63 ms, name ranking 5, sorting and formatting 10, writing 4 | run-e4fce5920002 | 0.0728 | 63.8 | 67.83 |
| 2 | Target parsing first: AVX2 space scan, SWAR latency parse, smaller 128 KiB blocks. Replace the full sort with a counting sort by endpoint plus nth-element selection. Rank names through one merged hash table, and hand-format the numbers. Laptop: 69 ms to 34 ms | | | | |

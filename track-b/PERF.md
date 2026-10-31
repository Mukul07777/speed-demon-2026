# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2,200 | 0 |
| 1 | Rust rewrite built around the queries being known first: only query terms are indexed, at document level, with no positions. A term in 1/32 of a 12 MiB sample gets a bitmap, the rest get sorted lists. Phrases start from their rarest term and check candidates against the mapped text with an AVX2 search. Laptop: 180 ms, 131 MiB. Checked against the seed-1 hash, the worked example, and a brute-force answer on a mixed corpus | | | | |

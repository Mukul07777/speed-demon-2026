# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.35 | 2,200 | 0 |
| 1 | Rust rewrite built around the queries being known first: only query terms are indexed, at document level, with no positions. A term in 1/32 of a 12 MiB sample gets a bitmap, the rest get sorted lists. Phrases start from their rarest term and check candidates against the mapped text with an AVX2 search. Laptop: 180 ms, 131 MiB. Checked against the seed-1 hash, the worked example, and a brute-force answer on a mixed corpus | run-2791dad2f937 | 0.208 | 137.1 | 64.93 |
| 2 | Timed each phase: scan 99 ms of 180, answers 40, build 16, sample 8, query parsing 8. Inside the scan, rdtsc gave populate 25 Mticks, tokenising 450, lookup 2,900, so the cost was 76 ticks per token in cache misses and mispredicted branches, not reading or parsing. Replaced the table and filter with a 64-bit hash in one-cache-line buckets. That made the scan slower, 129 ms: a 2 MiB region at an unaligned address holds no huge page, so every lookup paid a TLB miss. Aligning it gave 103 ms. Then hashed a whole batch first, prefetched 12 tokens ahead, and split hits into heavy and list arrays with no branch: lookup 1,300 Mticks, scan 57 ms. Sample cut from 12 MiB to 3 MiB, queries parsed in one pass, the build lost an indirect call and a branch on every row. Compiler flags gave nothing here, so x86-64-v3 stays. Laptop: 121 ms, 126 MiB | | | | |

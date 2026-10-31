# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only, in two phases on one pool of threads. The design comes from profiling: see `PERF.md` for what each step bought.

**Parse.** The file is memory-mapped. A worker claims 1 MiB and maps it in with one `MADV_POPULATE_READ` call, because letting 12 threads fault pages in one at a time cost more than parsing them. It then works through 64 KiB blocks in two passes. The first finds every newline. Lines are longer than a 32-byte AVX2 step, so each step holds at most one newline and one unconditional store records it. The second pass reads each line from both ends with no branch that depends on the data: hour digits at a fixed offset, method length with a SWAR space search, and the latency back from the newline with SWAR arithmetic. Only about 1% of lines take a rare path.

Endpoints live in one shared open-addressed table that fills in the first moments and is read-only after that. A name is identified by a 64-bit hash alone (53 bits compared, the same documented shortcut Tracks B and C take; the chance that two of the 1,842 names collide is about 2e-10). The hash covers the first and last 32 bytes under an AVX2 mask, so any name up to 64 bytes needs no length-dependent loop, and the few longer names also fold in their middle bytes. A slot holds the hash tag and the endpoint id, and the name text is copied once, on first sight, for the output. Each line becomes a 4-byte row (endpoint, error bit, 20-bit latency) in a region reserved for its hour, and the 1% of latencies that need more bits take a second slot.

**Summarise.** Workers take one hour at a time. Each hour is radix sorted by latency inside its own pages, then dealt out by endpoint. Dealing keeps rows in order, so every endpoint comes out sorted and a percentile is one array read. The 1% of latencies too long for a row are sorted aside and follow the rest. The hour's memory goes back to the kernel at once, so rows and output never add up. Each output line is built with branch-free number formatting. Whichever worker finds the next hour ready writes it, so the single-threaded write overlaps the parallel work.

## Build

The binary is linked statically (`+crt-static`), which removes dynamic loading from the start-up the timer includes.

Threads are started one by one from the main thread. Starting them as a binary tree gave no measurable gain (see `PERF.md`).

## Dependencies

None.

## With another 24 hours

Memory is near its floor for this design: the peak is the end of parsing, when all rows (about 12.5 MB) are live, so summary scratch is not the limit (see `PERF.md` step 9). Smaller rows are the only lever left. Try profile-guided optimisation at build time, and replace selection on the largest groups with a histogram.

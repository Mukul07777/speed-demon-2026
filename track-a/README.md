# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only, in two phases on one pool of threads. The design comes from profiling: see `PERF.md` for what each step bought.

**Parse.** The file is memory-mapped. A worker claims 1 MiB and maps it in with one `MADV_POPULATE_READ` call, because letting 12 threads fault pages in one at a time cost more than parsing them. It then works through 64 KiB blocks in two passes. The first finds every newline. Lines are longer than a 32-byte AVX2 step, so each step holds at most one newline and one unconditional store records it. The second pass reads each line from both ends with no branch that depends on the data: hour digits at a fixed offset, method length with a SWAR space search, and the latency back from the newline with SWAR arithmetic. Only about 1% of lines take a rare path.

Endpoints live in one shared open-addressed table that fills in the first moments and is read-only after that. A lookup compares the first and last 32 bytes of the name with one AVX2 test, which settles any name up to 64 bytes without a length-dependent loop. Each line becomes a 4-byte row (endpoint, error bit, 20-bit latency) in a region reserved for its hour, and the 1% of latencies that need more bits take a second slot.

**Summarise.** Workers take one hour at a time. Each hour is radix sorted by latency inside its own pages, then dealt out by endpoint. Dealing keeps rows in order, so every endpoint comes out sorted and a percentile is one array read. The 1% of latencies too long for a row are sorted aside and follow the rest. The hour's memory goes back to the kernel at once, so rows and output never add up. Each output line is built with branch-free number formatting. Whichever worker finds the next hour ready writes it, so the single-threaded write overlaps the parallel work.

## Build

The binary is linked statically (`+crt-static`), which removes dynamic loading from the start-up the timer includes.

## Dependencies

None.

## With another 24 hours

Spawn the threads as a tree, try profile-guided optimisation at build time, and replace selection on the largest groups with a histogram.

# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only, on one pool of threads (the main thread works too). Parsing and summarising overlap: there is no barrier between them. The design comes from profiling: `PERF.md` has every step with its measurements, including the failures.

**Parse.** The file is memory-mapped, with one spare zero page after it so that wide loads near the end stay in bounds. Workers claim 1 MiB at a time and let pages fault in on demand (populating each claim first was faster on the laptop but not on the scoring box). Each claim is read in 64 KiB blocks, in two passes, and the next block is prefetched while the current one is parsed:

1. AVX2 finds every newline. Lines are longer than a 32-byte step, so each step holds at most one newline, and one unconditional store records it.
2. Each line is read from both ends with no branch that depends on the data. The hour comes from fixed offsets, the method length from a SWAR space search, and the latency is read back from the newline with SWAR arithmetic. Only latencies of eight digits or more take a rare path.

When a claim is done, its file pages are dropped with `MADV_DONTNEED`. Left mapped, they would be torn down by one thread at exit.

**Endpoints.** One shared open-addressed table maps a name to its id. It fills in the first moments of the run and is read-only after that. A name is identified by a 64-bit hash alone (53 bits compared). This is the same deliberate shortcut Tracks B and C take: the chance that two of the 1,842 names collide is about 2e-10. The hash is VAES rounds over the masked first and last 32 bytes, so a name up to 64 bytes needs no length-dependent loop. The few longer names also fold in their middle bytes. A new name is inserted under a lock.

**Rows.** Each line becomes a 4-byte row: endpoint id, error bit and a 20-bit latency. The 1% of latencies that need more bits take a second row. Rows go into a fixed area reserved for their hour, in chunks of 256 rows that each worker claims for itself.

**Hours are summarised while parsing.** The lines are an interleaving of time-ordered streams, so when the parsed prefix of the file holds a line of hour h + 2, no later claim should hold one of hour h. After each claim a worker records the lowest and highest hour it saw, and hour h is then sealed. Any thread that is free takes the next sealed hour. Only a few hours of rows are alive at once, not all 72, which is how the peak fell from about 20 MB to 16 MB.

This is a bet on the input's order, and it is checked: if a claim turns out to hold a line of an hour that was already sealed, `dirty` is set and the whole job reruns with nothing sealed before the end of the file. The output is exact for any line order (tested on shuffled, reversed and late-line files).

**Summarise an hour.** In the hour's own pages:

- two 10-bit radix passes sort the rows by latency, and the escaped rows are sorted aside and follow the rest;
- the sorted rows are dealt out by endpoint, which keeps each endpoint's latencies in order, so a percentile is one array read;
- the output text is built over the rows, with branch-free number formatting.

Nothing is allocated per hour. Once every earlier hour's text exists, an hour's offset in the file is known. Whichever thread finds it ready writes it with `write_all_at` at that offset and gives the hour's pages back. Writes therefore run on several threads at once and overlap the parsing. The output file is on tmpfs, which counts against memory, so releasing rows as the file grows keeps the peak down.

## Build

The binary is linked statically (`+crt-static`), which removes dynamic loading from the start-up the timer includes. `+aes,+vaes` are enabled for the name hash; the scoring CPU has both.

## Tried and reverted

Each of these is in `PERF.md` with its numbers; none is in `src`.

- Reading with `pread` instead of a mapping, a futex barrier, pinning threads, large-slice page release, a reused sort scratch, an in-place summary, dropping the radix sort, a line cut to S 353: ties or losses, usually on the box even where the laptop showed a gain.
- Sealing hours by timestamp: the ceiling is about 0.2 ms by measurement, and hour 71 cannot seal before the end of the file.
- A 20 microsecond sleep instead of yielding at the end of the parse (`run-0a2da0f0982f`, 85.29): less memory, but 8% slower.

## Dependencies

None.

## With another 24 hours

Parsing is a read floor (about 6 ms for the file) plus about 8 ms of line work spread over the name lookup, row stores and decoding, and no single step is large (`PERF.md` step 11 and the removal profile). The peak of 16 MB is about 12 hours of rows, sort scratch and text plus the output on tmpfs. The end-of-file tail is only 3 to 9 hours, so splitting it across threads was tried and lost. The next try would be a cheaper per-line path through the name table, for example a small exact table for the most common names in front of the hash table.

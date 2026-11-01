# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only, in two phases on one pool of threads (the main thread works too). The design comes from profiling: `PERF.md` has every step with its measurements, including the failures.

**Parse.** The file is memory-mapped, with one spare zero page after it so that wide loads near the end stay in bounds. Workers claim 1 MiB at a time and let pages fault in on demand. Populating each claim first was faster on the laptop but not on the scoring box. Each claim is read in 64 KiB blocks, in two passes:

1. AVX2 finds every newline. Lines are longer than a 32-byte step, so each step holds at most one newline, and one unconditional store records it.
2. Each line is read from both ends with no branch that depends on the data. The hour comes from fixed offsets, the method length from a SWAR space search, and the latency is read back from the newline with SWAR arithmetic. Only latencies of eight digits or more take a rare path.

When a claim is done, its file pages are dropped with `MADV_DONTNEED`. Left mapped, they would be torn down by one thread at exit.

**Endpoints.** One shared open-addressed table maps a name to its id. It fills in the first moments of the run and is read-only after that. A name is identified by a 64-bit hash alone (53 bits compared). This is the same deliberate shortcut Tracks B and C take: the chance that two of the 1,842 names collide is about 2e-10. The hash is three VAES rounds over the masked first and last 32 bytes, so a name up to 64 bytes needs no length-dependent loop. The few longer names also fold in their middle bytes. A new name is inserted under a lock.

**Rows.** Each line becomes a 4-byte row: endpoint id, error bit and a 20-bit latency. The 1% of latencies that need more bits take a second row. Rows go into a fixed area reserved for their hour, in chunks of 256 rows that each worker claims for itself.

**Summarise.** After one barrier, workers take one hour at a time. In the hour's own pages:

- two 10-bit radix passes sort the rows by latency, and the escaped rows are sorted aside and follow the rest;
- the sorted rows are dealt out by endpoint, which keeps each endpoint's latencies in order, so a percentile is one array read;
- the output text is built over the rows, with branch-free number formatting.

Nothing is allocated per hour. Once every earlier hour's text exists, an hour's offset in the file is known. Whichever thread finds it ready writes it with `pwrite` and gives the hour's pages back. Writes therefore run on several threads at once and overlap the summaries, and memory is released as the output file (on tmpfs, which also counts against memory) grows.

## Build

The binary is linked statically (`+crt-static`), which removes dynamic loading from the start-up the timer includes. `+aes,+vaes` are enabled for the name hash; the scoring CPU has both.

## Dependencies

None.

## With another 24 hours

Parsing is the newline-scan floor (about 6 ms of reading) plus about 7.6 ms of line work spread over name lookup, row stores and decoding, and no single step is large (`PERF.md` step 11). The memory floor of this design is the 12.5 MB of rows live at the end of parsing. Teams at 16 MB must summarise during the parse. Our version of that (step 23) was exact but did not pay for its lines. The next try would be a cheaper per-line path through the name table, for example a small exact table for the most common names in front of the hash table.

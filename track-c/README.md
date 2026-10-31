# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Rust, standard library only. The design comes from measuring the data first: see `PERF.md`.

**Parallel start.** The file can be split anywhere without reading from its start. The spec guarantees that inside a note an LF is followed only by a lowercase word or a quote, so an LF followed by a digit or a BOM always begins a record. Each worker claims 1 MiB, maps it in with `MADV_POPULATE_READ`, finds the first record, and reads every record that starts inside its claim, running past the end if the last one does.

**Structure.** For each 64 bytes, AVX2 finds quotes, commas and LFs. A carry-less multiply turns the quote bits into an inside-quotes mask. A `""` toggles twice, so escapes need no special case. Every record then has exactly five structural bytes, which makes reading a record a fixed sequence of steps.

**Fields.** About 23% of records hold an escape and field lengths vary, so branching on either would mispredict constantly. Numbers are converted eight digits at a time with SWAR arithmetic, shifted according to their length. The BOM skip, the CR and the sign are all arithmetic. A field's value length comes from its raw length and its quote count: raw length L with q quote bytes has L - q/2 - 1 value bytes, or L if q is 0. Quote counts come from a rank query on the saved quote masks.

**Distinct names.** Names are compared by their raw bytes, because every non-empty name is quoted, so equal raw bytes mean equal values. Names are drawn uniformly from 100,000 values, so every lookup misses the cache. Three things answer that. A batch prefetches hash slots, then entries, and only then touches them. An entry keeps the name and a 256-bit set of its countries in one cache line, so one miss pays for both. And `fetch_or` reports whether this worker was first to set a bit, so the distinct count is a running total with no final scan. The table takes no lock: an entry is complete before a compare-and-swap publishes it.

## Dependencies

None.

## With another 24 hours

Try partitioning each batch by name hash, so entries could be updated without atomics, and try batching the region lookups the same way as the name lookups.

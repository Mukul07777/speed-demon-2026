# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Rust, standard library only. The design comes from measuring the data first: see `PERF.md`.

**Parallel start.** The file can be split anywhere without reading from its start. The spec guarantees that inside a note an LF is followed only by a lowercase word or a quote, so an LF followed by a digit or a BOM always begins a record. Each worker claims 1 MiB, maps it in with `MADV_POPULATE_READ`, finds the first record, and reads every record that starts inside its claim, running past the end if the last one does.

**Structure.** For each 64 bytes, AVX2 finds quotes, commas and LFs. A carry-less multiply turns the quote bits into an inside-quotes mask. A `""` toggles twice, so escapes need no special case. Every record then has exactly five structural bytes, which makes reading a record a fixed sequence of steps.

**Fields.** About 23% of records hold an escape and field lengths vary, so branching on either would mispredict constantly. Numbers are shifted according to their length so that the bytes below the digits read as leading zeros. The region is converted with SWAR arithmetic as the record is read, because its table entry is prefetched at once. The id and the amount are only stored as eight-digit words: after the batch, `decode_digits` converts four of them at a time with `vpmaddubsw`, `vpmaddwd` and a pack, which costs a fraction of a scalar multiply chain per number. The BOM skip, the CR and the sign are all arithmetic. A field's value length comes from its raw length and its quote count: raw length L with q quote bytes has L - q/2 - 1 value bytes, or L if q is 0. A name's quote count comes free from the 32 bytes the hash has already loaded, so only the end of the note needs a rank query on the saved quote masks. A record's first rank is the previous record's last, because only names and notes hold quotes.

**Distinct names.** Names are compared by their raw bytes, because every non-empty name is quoted, so equal raw bytes mean equal values. Names are drawn uniformly from 100,000 values, so every lookup misses the cache. A name is reduced to a 64-bit hash and looked up in a table of 64-byte buckets, prefetched a batch ahead, so a lookup is one cache line and one AVX2 compare. About 5% of names find their bucket full and spill to the next one, which a cold function handles. Comparing hashes instead of names is the one deliberate shortcut in this program: two of the 100,000 names sharing a hash has odds of about 3 in 10 billion, far below the chance of a hardware fault, and the saving is a memory access per record. Names are added with a compare-and-swap on each slot, so no lock is needed.

Each record sets its name's bit in its country's row of one shared bit matrix, and popcounts of the rows give the distinct names. Names get dense numbers, kept in the bucket next to their hashes, so the matrix is 4 MiB and sits on huge pages. A locked update of a cold line stalls the core, so a batch prefetches its lines for write first and updates them in a second pass.

## Dependencies

None.

## With another 24 hours

Decode the region in the vector batch too (it cost about 1.5% when tried, because the prefetch then sits right before the read), and give the 5% of names that overflow their bucket a second probe without a cache miss, for example by making buckets 128 bytes. A smaller bit matrix by partitioning records on `id >> 16` is still untested on the real box.

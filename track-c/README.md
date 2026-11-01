# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

Rust, standard library only. The design comes from measuring the data first. `PERF.md` has every step, including the failures.

**Threads.** `main` maps both files and starts one worker per CPU as a tree. Each worker zeroes one huge page of the shared tables. The workers then parse `regions.csv` together into a table from region id to country group, wait at a single barrier, and parse `records.csv` in 1 MiB claims. Each worker keeps its own statistics per group, and these are merged at the end. Claims are released with `MADV_DONTNEED` as they finish, so the kernel does not tear down the whole mapping on one thread at exit.

**Parallel start.** The file can be split anywhere without reading from its start. The spec guarantees that inside a note an LF is followed only by a lowercase word or a quote. An LF followed by a digit or a BOM therefore always begins a record.

**Structure.** For each 64 bytes, AVX2 finds quotes, commas and LFs. A carry-less multiply turns the quote bits into an inside-quotes mask. A `""` toggles twice, so escapes need no special case. Every record then has exactly five structural bytes, whose positions are stored with unconditional writes.

**Fields.** About 23% of records hold an escape and field lengths vary, so any branch on either would mispredict constantly. The program has none:

- The digits of the id, region and amount are shifted into eight-byte words, so the bytes below the digits read as leading zeros. `decode_digits` converts them four records at a time with `vpmaddubsw`, `vpmaddwd` and a pack.
- The BOM, the CR and the sign are handled by arithmetic.
- A field with raw length L and q quote bytes has a value of L - q/2 - 1 bytes, or L when q is 0. A name's quote count comes from the 32 bytes its hash has already loaded, so only the note needs a rank query on the saved quote masks.

**Names.** Names are drawn uniformly from 100,000 values, so every lookup misses the cache. A batch therefore runs in passes:

1. Read the fields, then prefetch the record's region entry and name bucket.
2. Read both, update the statistics, and number the name. Then prefetch the bit-matrix line for write.
3. Set the bit with `fetch_or`.

A name is a 64-bit AES hash. Its top 15 bits select a 64-byte bucket of eight u64 slots. Each slot holds a 47-bit tag above the name's dense 17-bit number, so a single compare-and-swap claims a slot together with its number, and no lock is needed. Workers reserve numbers in blocks of 32, so a first appearance does not touch a shared counter.

Comparing hashes instead of names is a deliberate shortcut. Two of the 100,000 names would have to match on all 62 bits, at odds of about 1 in a billion, and the shortcut saves a memory access per record.

**Distinct names.** One shared bit matrix has a row per country group, indexed by the name's number. It is 256 × 2^17 bits, or 4 MiB, on huge pages. Popcounts of the rows give the distinct-name counts.

## Dependencies

None.

## With another 24 hours

Interning a name is the largest remaining step, because the shared name table misses the cache on every record. An unlocked probe, which gives wrong answers, showed a ceiling of only about 2 ms for removing the lock on the bit matrix. A real gain needs a layout that stops threads sharing these tables without 12 private 4 MiB copies.

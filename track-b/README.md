# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only. The queries can be read before the corpus, so the program answers those 50,000 queries instead of building a general index. `PERF.md` has every step with its measurements, including the failures.

**Terms.** Each thread parses a share of `queries.txt` and interns its terms into one shared table of 2^15 buckets. A bucket is one cache line with 7 tags of 32 bits and 8 actions, on a huge page. A token is hashed with two AES rounds over its first 16 bytes ORed with `0x20`. Every token byte is a letter or a digit, so `| 0x20` lowercases letters and leaves digits as they are, with no check per byte. A lookup is one 256-bit compare, and a token that is not in the table takes the same path as one that is. Tokens are compared by 47 hash bits (15 pick the bucket, 32 are the tag). This is a deliberate shortcut: the odds of a false match are about 3 in 10,000 over the whole corpus.

**Scan.** The corpus is mapped and split into 256 KiB claims. Each claim starts on a document whose id is a multiple of 64, so no two threads ever write the same bitmap word. A batch of 16 KiB goes through these steps:

1. AVX2 marks token bytes and newlines, 64 bytes at a time. Token starts and ends are written with unconditional stores.
2. Tokens are hashed two at a time with VAES.
3. One loop looks up each token, with its bucket prefetched 20 tokens ahead. It stores one action per token, then packs the hits 8 tokens at a time with one AVX2 `permutevar8x32` and a 256-entry table, with no branch and no scalar store per token, into bitmap hits, list hits and phrase anchors. A bitmap hit only sets a bit in a small row for the current document, which stays in L1; every 64 documents a 64x64 bit transpose moves those rows into the phase's bitmaps.

A small sample (12 claims) chooses which terms get bitmaps: those in at least 1/32 of the documents. List terms write 4-byte rows into chunks per shard, and each chunk records the claim that wrote it. Taking the chunks in claim order therefore gives sorted document lists without sorting.

**Phrases.** No positions are stored. Every phrase is anchored on its rarest word in the sample. When an anchor appears in the scan, the hashes of the tokens around it are compared with the phrase. A match writes a row for a pseudo-term that stands for the phrase, so a phrase query becomes a one-term lookup. A batch ends on a document boundary, and each document's id token gets hash 0, so a check never runs into the next document.

**Phases and answers.** The corpus is scanned in 5 phases of 50 MiB. After each phase:

- its rows become lists of 16-bit offsets from the phase's first document, and the row pages are reused;
- every query is evaluated on that phase alone, strictly left to right, and the result is added to the query's tally (count, id sum, ten smallest ids).

As a result the whole index never exists at once, and peak memory is about 42 MB. A BOOL step uses whichever form suits its operands:

- a branch-free search when one list is much shorter than the other;
- a scratch bitmap of marks when the lists are of similar length;
- a bit test between a list and a bitmap;
- word operations between two bitmaps.

A bitmap result gets its count and id sum from nibble lookups, 32 bytes at a time.

The answers of phase k-1 run on whichever threads have run out of scan claims in phase k, which fills the wait at the barrier after each scan. Barriers are one futex word (spin 3000 pauses, then `FUTEX_WAIT`), because the standard barrier took 1 to 4 ms to wake its threads. `PR_SET_THP_DISABLE` is set at the start, after the bucket table has been touched, so that table alone keeps its huge page.

## Tried and reverted

Each is in `PERF.md` with its numbers; none is in `src`.

- A 26 MiB peak (100 claims per phase and smaller buffers): memory under 44 MB earns nothing for this track, and the time cost made it lose (`run-004a611bfcd9`, 77.15 against 78.18).
- Spreading the previous phase's answers evenly through the next scan, a read-ahead prefetch, a pre-fault with `MADV_POPULATE_WRITE`, a trailing-OR shortcut, a SIMD list AND, an interleaved OR union, profile-guided optimisation: ties or losses.
- Exact 8-byte keys for short tokens: correct, but 73.7 to 82.3 ms best, so slower.

## Dependencies

None.

## With another 24 hours

The scan is about 60% of the time and is bound by instruction count, not by memory latency. A stage floor measured on the laptop (read 6.3 ms, tokenise 8.9, hash 14.2, lookup about 25, against a scan of 42 ms) says the lookup is the target. An exact short-key table was tried and lost, so the next step would be a cheaper path per token that does not add passes, for example tokenising and hashing fused in one loop.

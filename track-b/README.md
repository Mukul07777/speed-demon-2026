# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only. The queries can be read before the corpus, so the program answers those 50,000 queries instead of building a general index. `PERF.md` has every step with its measurements, including the failures.

**Terms.** Each thread parses a share of `queries.txt` and interns its terms into one shared table of 2^15 buckets. A bucket is one cache line with 7 tags of 32 bits and 8 actions, on a huge page. A token is hashed with two AES rounds over its first 16 bytes ORed with `0x20`. Every token byte is a letter or a digit, so `| 0x20` lowercases letters and leaves digits as they are, with no check per byte. A lookup is one 256-bit compare, and a token that is not in the table takes the same path as one that is. Tokens are compared by 47 hash bits (15 pick the bucket, 32 are the tag). This is a deliberate shortcut: the odds of a false match are about 3 in 10,000 over the whole corpus.

**Scan.** The corpus is mapped and split into 256 KiB claims. Each claim starts on a document whose id is a multiple of 64, so no two threads ever write the same bitmap word. A batch of 16 KiB goes through these steps:

1. AVX2 marks token bytes and newlines, 64 bytes at a time. Token starts and ends are written with unconditional stores.
2. Tokens are hashed two at a time with VAES.
3. One loop looks up each token, with its bucket prefetched 20 tokens ahead. It splits hits with no branch into bitmap hits, list hits and phrase anchors.

A small sample (12 claims) chooses which terms get bitmaps: those in at least 1/32 of the documents. List terms write 4-byte rows into chunks per shard, and each chunk records the claim that wrote it. Taking the chunks in claim order therefore gives sorted document lists without sorting.

**Phrases.** No positions are stored. Every phrase is anchored on its rarest word in the sample. When an anchor appears in the scan, the hashes of the tokens around it are compared with the phrase. A match writes a row for a pseudo-term that stands for the phrase, so a phrase query becomes a one-term lookup. A batch ends on a document boundary, and each document's id token gets hash 0, so a check never runs into the next document.

**Phases and answers.** The corpus is scanned in 5 phases of 50 MiB. After each phase:

- its rows become lists of 16-bit offsets from the phase's first document, and the row pages are reused;
- every query is evaluated on that phase alone, strictly left to right, and the result is added to the query's tally (count, id sum, ten smallest ids).

As a result the whole index never exists at once, and peak memory is about 41 MB. A BOOL step uses whichever form suits its operands:

- a branch-free search when one list is much shorter than the other;
- a scratch bitmap of marks when the lists are of similar length;
- a bit test between a list and a bitmap;
- word operations between two bitmaps.

A bitmap result gets its count and id sum from nibble lookups, 32 bytes at a time.

The answers of phase k-1 run on whichever threads have run out of scan claims in phase k, which fills the wait at the barrier after each scan. `PR_SET_THP_DISABLE` is set at the start, after the bucket table has been touched, so that table alone keeps its huge page.

## Dependencies

None.

## With another 24 hours

The scan is about 60% of the time and is bound by instruction count, not by memory latency. The next step would be a cheaper path per token: tokenising and hashing fused in one pass, or an exact short-key table for the most common terms. After that, a SIMD union for OR steps.

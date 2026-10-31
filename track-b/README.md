# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only. The queries are readable from the start, so this answers those 50,000 queries rather than indexing the corpus in general.

**Terms.** Every query term goes into one table first. A token that is not in the table is dropped after one hash and a bit test, whatever else the corpus holds. Tokenising uses AVX2 masks for token bytes. Every byte of a token is a letter or digit, and `| 0x20` lowers letters and leaves digits alone, so case folding needs no per-byte check, and non-ASCII bytes are separators for free.

**Index.** Everything is at document level, and no positions are stored. Each worker first reads one 1 MiB claim and only counts terms. A term that reaches 1/32 of the sampled documents gets a bitmap, and everything else gets a sorted list of document ids. Claims start on document ids that are multiples of 64, so no two workers ever share a word of a bitmap and setting a bit needs no atomics. List terms write 4-byte rows into chunks per shard. Each chunk records which claim wrote it, and a claim writes in document order, so taking chunks in claim order gives sorted lists with no sorting of rows.

**BOOL.** Evaluated strictly left to right, as the spec requires, on whichever of list or bitmap suits each step: galloping intersection and difference between lists, a bit test between a list and a bitmap, and word-wide operations between bitmaps. The bitmap of a term is used in place until a step has to change it. A result held as a bitmap gets its count and id sum from popcounts, one for each bit of the position.

**PHRASE.** A phrase starts from its rarest term, and the spec guarantees that nearly every phrase has a rare one. Candidates are filtered by the phrase's other terms, then checked against the mapped text: an AVX2 search for the first and last byte of the term finds its occurrences, and the check steps out token by token in both directions. Separators never need handling because stepping skips all of them, which is also what lets a phrase cross a sentence boundary.

## Dependencies

None.

## With another 24 hours

Stop the sample reading its claims twice, shrink rows and postings so they do not coexist at full size, and lower the bitmap threshold where that saves rows. Then use a smaller table for the most common tokens, so that rare ones cost less in the cache.

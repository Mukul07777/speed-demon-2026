# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

Rust, standard library only. The queries are readable from the start, so this answers those 50,000 queries rather than indexing the corpus in general.

**Terms.** Every query term goes into one table first. A token is hashed to 64 bits and looked up in a table of 64-byte buckets, each holding five hashes and their actions, so a lookup is one cache line and one AVX2 compare. Comparing hashes instead of tokens is a deliberate shortcut: of 600,000 tokens and 80,000 terms, two sharing a hash has odds of about 3 in a billion. Tokenising uses AVX2 masks for token bytes. Every byte of a token is a letter or digit, and `| 0x20` lowers letters and leaves digits alone, so case folding needs no per-byte check, and non-ASCII bytes are separators for free.

**Lookup.** This was 85% of the scan, spent on cache misses and mispredicted branches rather than on any real work. So a batch of text is hashed first, each bucket is prefetched 12 tokens before its lookup, and a token missing from the table takes the same path as one that is present. Hits are split into heavy and list arrays with no branch, and each array is applied in its own pass. The table sits on a huge page, which must be aligned or it does not exist: an unaligned 2 MiB region made the scan 25% slower.

**Index.** Everything is at document level, and no positions are stored. First a 3 MiB sample only counts terms. A term that reaches 1/32 of the sampled documents gets a bitmap, and everything else gets a sorted list of document ids. Claims start on document ids that are multiples of 64, so no two workers ever share a word of a bitmap and setting a bit needs no atomics. List terms write 4-byte rows into chunks per shard. Each chunk records which claim wrote it, and a claim writes in document order, so taking chunks in claim order gives sorted lists with no sorting of rows.

**Phases.** Rows and postings each came to about 50 MB, so rows only live for one phase, 32 MiB of text. After each phase the rows become postings and their pages are used again. A posting is a 16-bit offset from the phase's first document, in a part chained to the term's previous part. Those parts are widened to a plain sorted list only when a query needs it, which costs little beside the work the query does with the list.

**BOOL.** Evaluated strictly left to right, as the spec requires, on whichever of list or bitmap suits each step: galloping intersection and difference between lists, a bit test between a list and a bitmap, and word-wide operations between bitmaps. The bitmap of a term is used in place until a step has to change it. A result held as a bitmap gets its count and id sum from popcounts, one for each bit of the position.

**PHRASE.** Checked during the scan, because that is when a document's tokens and their hashes are already in memory. The sample names each phrase's rarest word as its anchor. When an anchor appears, the neighbouring hashes are compared with the phrase, and a match writes a row for a pseudo-term that stands for the phrase, so a phrase query is a one-term lookup afterwards. Separators need no handling, because only tokens are in the array, which is also what lets a phrase cross a sentence boundary. A batch stops at a document boundary so that a phrase always has its tokens, and each document's id token gets hash 0, so no check can run into the next document. The first version searched each candidate's text after the scan, and 184,000 of 214,000 candidates failed only after reading a whole document.

## Dependencies

None.

## With another 24 hours

Tokenising is now the largest part of the scan, so finish token ends and starts in one array. Then look at the 230 ms of CPU the BOOL queries take.

**Query parsing and huge pages.** The query file is split at line starts; threads hash their share's words in parallel and the main thread interns each share in file order, prefetching buckets, because interning touches shared state and must keep query order. Transparent huge pages are forbidden from the start with `PR_SET_THP_DISABLE`, after the bucket table is touched so that table alone keeps its huge page. On a kernel with THP `always` every other region would otherwise be backed 2 MiB at a time, which charges memory for pages that are mostly empty. The effect on the scoring box is unmeasured at the time of writing; the laptop showed 3 MiB less and no slowdown.

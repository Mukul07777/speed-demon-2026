# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Rust, standard library only. Node ids are a random relabelling of a road map, so in file order every step of a search would be a cache miss. Most of the program exists to undo that.

**Parse.** Workers claim 1 MiB of the mapped file. A line is at most 27 bytes, so one AVX2 load holds all three separators, and digits convert with SWAR arithmetic. Edges go into shards by tail node, so no worker needs a shared counter.

**Compact edges.** An edge is 4 bytes: 21 bits of head and 11 of weight. Only about 3,500 of the 5,000,000 weights need more, so those edges sit in a sorted side list and their tails carry a flag.

**Renumber.** Workers grow cells of up to 1024 nodes over the forward graph with a breadth-first search, claiming nodes atomically. The end of a cell's queue seeds the next cell, so neighbouring cells get neighbouring numbers. In the old numbering a node's entry, its edges and its label are three cache misses, so the queue is used to prefetch each of them in turn before it is needed. Rewriting the edges works in small batches for the same reason. Reverse edges then come straight from the renumbered forward edges with atomic counters: one pass counts while rewriting, and one fills in.

**Search.** Each query runs a bidirectional Dijkstra, one thread per query, always advancing the side with the fewer frontier entries. Settling a node checks the other side's tentative label, which is enough to catch every meeting point. The search stops when the two frontier minimums add up to the best distance found. If a side runs dry it has settled everything it can reach, which answers unreachable targets at once.

Light edges reach less than 2048 ahead, so the frontier is a circular window of exact-distance buckets, and popping is finding the next set bit. Heavy edges wait in a small heap outside the window. Distances sit in 64-node blocks, handed out when a search first reaches them, so a search only pays for what it touches. The graph does not fit in the cache, so a new label prefetches the edges that settling it will read.

## Dependencies

None.

## With another 24 hours

Store distances in 32 bits, with a 64-bit fallback for the few queries that cross a heavy edge. Then try contracting nodes of degree one and two, which shrinks the search for every query but costs build time.

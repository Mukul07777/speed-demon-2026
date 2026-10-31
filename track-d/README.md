# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Rust, standard library only. Node ids are a random relabelling of a road map, so in file order every step of a search would be a cache miss. Most of the program exists to undo that.

**Parse.** Workers claim 1 MiB of the mapped file. A line is at most 27 bytes, so one AVX2 load holds all three separators, and digits convert with SWAR arithmetic. Edges go into shards by tail node, so no worker needs a shared counter.

**Compact edges.** An edge is 4 bytes: 21 bits of head and 11 of weight. Only about 3,500 of the 5,000,000 weights need more, so those edges sit in a sorted side list and their tails carry a flag.

**Renumber.** Workers grow cells of up to 1024 nodes over the forward graph with a breadth-first search, claiming nodes atomically. What is left of a cell's queue seeds the next cell, so neighbouring cells get neighbouring numbers. Nodes close on the map end up close in memory, and the forward and backward adjacency arrays are rewritten in that order.

**Search.** Each query runs a bidirectional Dijkstra, one thread per query, always advancing the side with the smaller frontier. Settling a node checks the other side's tentative label, which is enough to catch every meeting point. The search stops when the two frontier minimums add up to the best distance found. If a side runs dry it has settled everything it can reach, which answers unreachable targets at once. Distances sit in blocks that are handed out when a search first reaches a stretch of the numbering, so a search only pays for what it touches.

## Dependencies

None.

## With another 24 hours

Replace the binary heap with a monotone radix heap, shrink the rows used while building, and try contracting nodes of degree one and two.

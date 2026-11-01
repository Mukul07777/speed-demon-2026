# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

Rust, standard library only. Node ids are a random relabelling of a road map, so in file order every step of a search would be a cache miss. Most of the program exists to undo that. All threads run the same phases, with a barrier between phases and jobs taken from one shared counter. `PERF.md` has every step, including the failures.

**Parse.** Workers claim 1 MiB of the mapped file. A line is at most 27 bytes, so one AVX2 load holds all three separators, and digits are converted with SWAR arithmetic. Edges go into rows sharded by tail node. Each worker owns a 64-row chunk per shard, so a row needs no shared counter. The shards are then built into a raw adjacency array, and their rows are released.

**Compact edges.** An edge is 4 bytes: 21 bits of head and 11 of weight. A node entry is `start << 5 | degree`. Only about 3,500 of the 5,000,000 weights need more than 11 bits. Those edges carry an all-ones weight, and the real weight is kept in a hash map keyed by (tail, head).

**Renumber.** Workers grow cells of up to 1024 nodes with a breadth-first search over the raw graph. They claim nodes by a CAS that writes the node's new position into its raw entry. The last nodes in a cell's queue seed the next cell, so neighbouring cells get neighbouring numbers. In the old numbering a node's entry and its edges are two cache misses, so the queue is used to prefetch both before they are needed.

A claimed node's edges are copied straight into the final forward array, so growing the cells and laying out the forward array are one pass. Next, the edge heads are translated to new ids in batches of 64, each batch prefetched first, while the in-degrees are counted atomically. The backward array is then filled from those counts. The raw graph is released once it is no longer needed, each worker giving back its own stretch of the region.

**Search.** Each query runs a bidirectional Dijkstra, one query per job. The search always advances the side with fewer live entries. Settling a node checks the other side's label, which catches every meeting point. The search stops when the two frontier minimums add up to the best distance found. If one side runs dry, it has settled everything it can reach, which answers an unreachable target at once.

Light edges reach less than 2048 ahead, so the frontier is a circular window of exact-distance buckets, and a pop scans for the next non-empty bucket. Heavy edges wait in a small heap outside the window. Distances are u32 values (wider ones go in a side map) in 64-node blocks, handed out when a search first reaches them, so a search pays only for what it touches. The graph does not fit in the cache, so each new label prefetches the edges that settling its node will read.

## Tried and reverted

Each is in `PERF.md` with its numbers; none is in `src`.

- Serving the backward search from the forward graph (a parallel array of reverse weights): 9 MB less memory, but the score tied (72.16 against 72.22) once lines and time were counted.
- A binary heap in place of the bucket queue (17 to 54% slower), dropping `MADV_POPULATE_READ` (slower on the box), profile-guided optimisation (under the 3% bar).

## Dependencies

None.

## With another 24 hours

Queries take about 100 cycles per settled node, bound by DRAM latency. Interleaving several searches, sorting the queries and pruning nodes with one neighbour were all tried and gave nothing. What remains is contracting nodes of degree two, which shrinks every search but costs build time and code.

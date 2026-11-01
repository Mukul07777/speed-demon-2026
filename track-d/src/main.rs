//! Speed Demon 2026, Track D (Traverse).
//!
//! Node ids are a random relabelling of a road map, so in file order every step of a search
//! is a cache miss. Most of the program therefore exists to undo that:
//!
//! Every thread runs `Run::work`, the same phases in turn, with a barrier between them
//! (`next_step`) and jobs claimed from one shared counter:
//!
//! 1. Parse 1 MiB claims of edges into rows, sharded by tail node. Each worker owns a
//!    64-row chunk per shard at a time, so a row costs no shared counter.
//! 2. Build the raw adjacency array from the shards, one shard per job, in any node order.
//! 3. Grow compact cells of nodes over that graph (`grow_cells`). Claiming a node writes its
//!    new position into its raw entry, and copies its edges straight into the final forward
//!    array, so nodes close on the map are close in memory and growing and laying out the
//!    forward array are one pass.
//! 4. Number the cells (`plan_layout`, one thread), then translate every forward edge head
//!    to its new id and count in-degrees, and fill the backward array from those counts.
//! 5. Answer the queries with a bidirectional Dijkstra, one query per job.
//!
//! An edge is 4 bytes: 21 bits of head and 11 of weight. A node entry is `start << 5 | degree`.
//! Only about 3,500 of the 5,000,000 weights need more than 11 bits, so those edges carry an
//! all-ones weight and the real one is looked up in a side table. Raw rows and the raw graph
//! are released with `MADV_DONTNEED` as soon as they are used, to keep the peak down.
use std::arch::x86_64::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex, OnceLock};
use std::thread;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn prctl(option: i32, arg: u64, a: u64, b: u64, c: u64) -> i32;
    fn madvise(addr: *const u8, len: usize, advice: i32) -> i32;
    fn mallopt(param: i32, value: i32) -> i32;
}
const PROT_RW: i32 = 3; // PROT_READ | PROT_WRITE
const MAP_ANON: i32 = 0x4022; // MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
const MAP_FILE: i32 = 0x12; // MAP_PRIVATE | MAP_FIXED
const PR_SET_THP_DISABLE: i32 = 41;
const MADV_DONTNEED: i32 = 4;
const MADV_POPULATE_READ: i32 = 22;

const CLAIM: usize = 1 << 20;
const SHARD_SHIFT: u32 = 13; // few enough shards that a worker's write positions stay in L1
const SHARD_NODES: usize = 1 << SHARD_SHIFT;
const SHARD_ROWS: usize = SHARD_NODES * 16; // a shard's nodes have fewer than 16 edges each on average
const CHUNK_ROWS: usize = 64;
const WEIGHT_BITS: u32 = 11;
const DEGREE_MASK: u32 = 31; // a node entry's low 5 bits; build asserts no degree is larger
const START_SHIFT: u32 = 5;
const WEIGHT_MASK: u32 = (1 << WEIGHT_BITS) - 1;
const CELL: usize = 1 << 10;
const EDGE_CLAIM: usize = 1 << 12; // edge slots a worker takes at a time for the nodes it claims
const SEEDS_KEPT: usize = 32; // of a finished cell's queue
const FAR_AHEAD: usize = 12; // queue entries ahead to prefetch a node
const NEAR_AHEAD: usize = 6; // queue entries ahead to prefetch its edges
const REWRITE_BATCH: usize = 64; // edges translated together, after prefetching
const BLOCK_SHIFT: u32 = 6; // distances live in blocks of this many neighbouring nodes
const BLOCK_NODES: usize = 1 << BLOCK_SHIFT;
const WINDOW: usize = 1 << WEIGHT_BITS; // edges with weights that fit reach less far ahead
const CLAIMED: u32 = 1 << 31; // a claimed node's entry is CLAIMED | its new id, above every raw entry (below 2^28)
const FLAG: u32 = 1 << 31; // a forward node entry's top bit: some arcs into the node have no reverse arc
const NO_REV: u16 = u16::MAX; // a reverse weight slot with no arc behind it
type Extras = Vec<(u32, u32, u32)>; // head, tail and weight of an arc whose head has no arc back
const UNREACHED: u64 = u64::MAX;
const NONE: u16 = u16::MAX; // a distance slot nothing has reached, or whose distance is in `Side::wide`

/// An edge on its way to an adjacency array: `other + 1` and weight, plus the low bits of
/// the node it belongs to, whose high bits are its shard. A packed value of 0 marks the
/// unused end of a chunk.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Row {
    packed: u32,
    low: u16,
}

/// Rows sharded by node, in chunks that workers claim from each shard's own reserved span.
struct Rows {
    rows: *mut Row,
    claimed: Vec<AtomicUsize>,
}

impl Rows {
    unsafe fn new(shards: usize) -> Rows {
        Rows {
            rows: reserve(shards * SHARD_ROWS * size_of::<Row>()).cast(),
            claimed: (0..shards).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    /// The rows of a shard, chunk by chunk, skipping each chunk's unused end.
    unsafe fn rows_of(&self, shard: usize) -> impl Iterator<Item = Row> + '_ {
        let claimed = self.claimed[shard].load(Relaxed);
        let rows = std::slice::from_raw_parts(self.rows.add(shard * SHARD_ROWS), claimed);
        (rows.chunks(CHUNK_ROWS))
            .flat_map(|chunk| chunk.iter().copied().take_while(|row| row.packed != 0))
    }
}

/// An adjacency array. Each node's entry is start << 5 | degree, so a node's edges take one
/// load to find, and shards may sit in any order.
struct Graph {
    nodes: *mut u32,
    edges: *mut u32,
    node_count: usize,
    edge_bytes: usize,
    next_edge: AtomicUsize,
}

impl Graph {
    unsafe fn new(node_count: usize, edge_count: usize) -> Graph {
        let edge_bytes = edge_count * 4 + 4096;
        Graph {
            nodes: reserve(node_count * 4).cast(),
            edges: reserve(edge_bytes).cast(),
            node_count,
            edge_bytes,
            next_edge: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    unsafe fn edges_at(&self, entry: u32) -> &[u32] {
        let start = ((entry << 1) >> (START_SHIFT + 1)) as usize;
        std::slice::from_raw_parts(self.edges.add(start), (entry & DEGREE_MASK) as usize)
    }

    /// Fills in one shard of nodes from its rows, then frees them. This graph's node count is
    /// a whole number of shards.
    unsafe fn build_shard(&self, rows: &Rows, shard: usize) {
        // Each node's degree, then the slot for its next edge.
        let mut cursor = [0u32; SHARD_NODES];
        for row in rows.rows_of(shard) {
            cursor[row.low as usize] += 1;
        }
        let total = cursor.iter().sum::<u32>() as usize;
        let mut next = self.next_edge.fetch_add(total, Relaxed) as u32;
        for (low, slot) in cursor.iter_mut().enumerate() {
            assert!(*slot <= DEGREE_MASK);
            *self.nodes.add((shard << SHARD_SHIFT) + low) = next << START_SHIFT | *slot;
            (*slot, next) = (next, next + *slot);
        }
        for row in rows.rows_of(shard) {
            let slot = &mut cursor[row.low as usize];
            // Undo the `other + 1` that kept a used row nonzero.
            *self.edges.add(*slot as usize) = row.packed - (1 << WEIGHT_BITS);
            *slot += 1;
        }
        let rows = rows.rows.add(shard * SHARD_ROWS);
        madvise(rows.cast(), SHARD_ROWS * size_of::<Row>(), MADV_DONTNEED);
    }
}

/// Anonymous memory that costs nothing until it is touched.
unsafe fn reserve(bytes: usize) -> *mut u8 {
    mmap(null_mut(), bytes, PROT_RW, MAP_ANON, -1, 0)
}

/// Maps a file over the start of an anonymous region that ends a page later, so that a load
/// that starts inside the file's last line may run past its end.
unsafe fn map_file(file: &File, len: usize) -> &'static [u8] {
    let base = reserve(len.div_ceil(4096) * 4096 + 4096);
    mmap(base, len, 1, MAP_FILE, file.as_raw_fd(), 0);
    std::slice::from_raw_parts(base, len)
}

/// The number in the `len` bytes before `end`. Up to eight digits take the same few
/// instructions whatever the length, so varying lengths do not mispredict.
#[inline(always)]
unsafe fn number_before(end: *const u8, len: usize) -> u64 {
    if len > 8 {
        return (0..len).fold(0, |v, i| v * 10 + (*end.sub(len - i) - b'0') as u64);
    }
    let v = (end.sub(8).cast::<u64>().read_unaligned() ^ 0x3030_3030_3030_3030) >> (64 - 8 * len)
        << (64 - 8 * len);
    let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

/// The weights that do not fit in an edge, by (tail, head). Parallel edges keep the lightest.
type Heavy = HashMap<(u32, u32), u32>;

struct Layout {
    queries: Vec<(u32, u32)>,
    heavy: Heavy, // by (tail, head); the backward side looks up (head, tail)
}

struct Run {
    text: &'static [u8], // the graph file, mapped with a page of slack after it
    nodes: usize,
    threads: usize,
    next_job: AtomicUsize,
    step: Barrier,
    rows: Rows,
    raw: Graph,
    forward: Graph,
    rev: *mut u16, // parallel to forward.edges: the lightest arc back along each edge, or NO_REV
    extras_raw: Mutex<Extras>,
    extras: OnceLock<Extras>, // arcs with no arc back, sorted by head
    heavy: Mutex<Vec<(u32, u32, u32)>>, // tail, head and weight
    next_id: AtomicU32,       // the first new id no worker has been given
    chunks: Mutex<Vec<(usize, usize)>>, // spans of forward edges that hold nodes' edges
    layout: OnceLock<Layout>,
    raw_queries: Vec<(u32, u32)>,
    answers: Mutex<Vec<u64>>,
}
unsafe impl Sync for Run {}

impl Run {
    unsafe fn parse(&self) {
        let mut cursors = vec![(null_mut(), null_mut()); self.rows.claimed.len()];
        let text = self.text.as_ptr();
        let digit_floor = _mm256_set1_epi8(b'0' as i8);
        let mut heavy = Vec::new();
        self.each_job(self.text.len().div_ceil(CLAIM), |job| {
            let claim = job * CLAIM;
            let end = (claim + CLAIM).min(self.text.len());
            madvise(text.add(claim), end - claim, MADV_POPULATE_READ);
            // A line belongs to the claim it starts in, so skip to the first newline from the
            // byte before the claim. The header belongs to nobody.
            let mut at = claim.saturating_sub(1);
            at += self.text[at..].iter().position(|&b| b == b'\n').unwrap() + 1;
            while at < end {
                // A line is at most 27 bytes, so one load holds all three of its separators.
                let line = text.add(at);
                let chunk = _mm256_loadu_si256(line.cast());
                let mut separators =
                    _mm256_movemask_epi8(_mm256_cmpgt_epi8(digit_floor, chunk)) as u32;
                let first = separators.trailing_zeros() as usize;
                separators &= separators - 1;
                let second = separators.trailing_zeros() as usize;
                separators &= separators - 1;
                let third = separators.trailing_zeros() as usize;
                let u = number_before(line.add(first), first) as usize;
                let v = number_before(line.add(second), second - first - 1) as usize;
                let w = number_before(line.add(third), third - second - 1);
                at += third + 1;
                if w >= WEIGHT_MASK as u64 {
                    heavy.push((u as u32, v as u32, w as u32));
                }
                // Append a row to its node's shard, at this worker's own cursor (the next free row
                // and the end of the chunk it owns), taking a new chunk when the last is full.
                let shard = u >> SHARD_SHIFT;
                let cursor = cursors.get_unchecked_mut(shard);
                if cursor.0 == cursor.1 {
                    let offset = self.rows.claimed[shard].fetch_add(CHUNK_ROWS, Relaxed);
                    cursor.0 = self.rows.rows.add(shard * SHARD_ROWS + offset);
                    cursor.1 = cursor.0.add(CHUNK_ROWS);
                }
                let packed = ((v as u32 + 1) << WEIGHT_BITS) | w.min(WEIGHT_MASK as u64) as u32;
                let low = (u & (SHARD_NODES - 1)) as u16;
                cursor.0.write_unaligned(Row { packed, low });
                cursor.0 = cursor.0.add(1);
            }
        });
        self.heavy.lock().unwrap().extend(heavy);
    }

    /// Claims jobs 0..count, one at a time, until none are left.
    fn each_job(&self, count: usize, job: impl FnMut(usize)) {
        (std::iter::repeat_with(|| self.next_job.fetch_add(1, Relaxed)))
            .take_while(|&next| next < count)
            .for_each(job);
    }

    /// Everyone waits; one worker does `alone` and resets the shared job counter.
    fn next_step(&self, alone: impl FnOnce()) {
        if self.step.wait().is_leader() {
            alone();
            self.next_job.store(0, Relaxed);
        }
        self.step.wait();
    }

    /// Grows cells: a breadth-first search from one seed that stops at CELL nodes. The end of
    /// its queue becomes seeds, newest first, so that the next cell starts next door and
    /// neighbouring cells land near each other in the numbering.
    ///
    /// With random ids a node's entry and its edges are two cache misses (the label lives in
    /// the entry). The queue says what comes next, so each is prefetched in turn before it is
    /// needed. A claimed node's edges are copied, still with old head ids, into this worker's
    /// own spans of the forward array, listed in the chunks returned. New ids come in windows
    /// of CELL, so a worker's cells are numbered together and its nodes need no later sorting.
    unsafe fn grow_cells(&self, worker: usize) {
        let (mut cursor, mut limit, mut from) = (0, 0, 0);
        let (mut id, mut window_end) = (0, 0);
        let (mut seeds, mut queue) = (Vec::<u32>::new(), Vec::<u32>::new());
        // The seeds that cells leave behind come first, then the worker's share of the nodes.
        let share = |worker: usize| (worker * self.nodes / self.threads) as u32;
        let mut scan = share(worker)..share(worker + 1);
        while let Some(seed) = seeds.pop().or_else(|| scan.next()) {
            if *self.raw.nodes.add(seed as usize) & CLAIMED != 0 {
                continue;
            }
            queue.clear();
            queue.push(seed);
            let (mut head, mut size) = (0, 0);
            while head < queue.len() && size < CELL {
                if let Some(&ahead) = queue.get(head + FAR_AHEAD) {
                    _mm_prefetch::<_MM_HINT_T0>(self.raw.nodes.add(ahead as usize).cast());
                }
                if let Some(&ahead) = queue.get(head + NEAR_AHEAD) {
                    // A label here makes a junk address, which a prefetch ignores.
                    let start = *self.raw.nodes.add(ahead as usize) >> START_SHIFT;
                    _mm_prefetch::<_MM_HINT_T0>(self.raw.edges.wrapping_add(start as usize).cast());
                }
                let node = queue[head] as usize;
                head += 1;
                // The raw entry gives way to the node's label, so claiming a node costs no
                // cache miss beyond reading its entry.
                let at = &*self.raw.nodes.add(node).cast::<AtomicU32>();
                let entry = at.load(Relaxed);
                if id == window_end {
                    id = self.next_id.fetch_add(CELL as u32, Relaxed);
                    window_end = id + CELL as u32;
                }
                let value = CLAIMED | id;
                if entry & CLAIMED != 0
                    || at.compare_exchange(entry, value, Relaxed, Relaxed).is_err()
                {
                    continue;
                }
                size += 1;
                id += 1;
                let neighbours = self.raw.edges_at(entry);
                if cursor + neighbours.len() > limit {
                    self.chunks.lock().unwrap().push((from, cursor));
                    from = self.forward.next_edge.fetch_add(EDGE_CLAIM, Relaxed);
                    (cursor, limit) = (from, from + EDGE_CLAIM);
                }
                let to = self.forward.edges.add(cursor);
                std::ptr::copy_nonoverlapping(neighbours.as_ptr(), to, neighbours.len());
                *self.forward.nodes.add(value as usize & !(CLAIMED as usize)) =
                    (cursor as u32) << START_SHIFT | neighbours.len() as u32;
                cursor += neighbours.len();
                queue.extend(neighbours.iter().map(|&edge| edge >> WEIGHT_BITS));
            }
            // The last SEEDS_KEPT queued nodes the cell did not reach become seeds.
            let kept = queue.len().saturating_sub(SEEDS_KEPT).max(head);
            seeds.extend_from_slice(&queue[kept..]);
        }
        self.chunks.lock().unwrap().push((from, cursor));
    }

    /// A node's new id, which growing the cells left in its raw entry.
    #[inline(always)]
    unsafe fn new_id(&self, old: usize) -> u32 {
        *self.raw.nodes.add(old) & !CLAIMED
    }

    fn plan_layout(&self) -> Layout {
        let map = |old: u32| unsafe { self.new_id(old as usize) };
        let pair = |&(s, t): &(u32, u32)| (map(s), map(t));
        let queries = self.raw_queries.iter().map(pair).collect();
        let mut heavy = Heavy::new();
        for &(u, v, w) in self.heavy.lock().unwrap().iter() {
            let known = heavy.entry((map(u), map(v))).or_insert(w);
            *known = (*known).min(w);
        }
        Layout { queries, heavy }
    }

    /// Translates every edge head, in place, to its new id, which is a random lookup per edge.
    /// Working in small batches lets each one prefetch all it will touch before it touches it.
    unsafe fn rewrite_forward(&self, (from, to): (usize, usize)) {
        let edges = std::slice::from_raw_parts_mut(self.forward.edges.add(from), to - from);
        for batch in edges.chunks_mut(REWRITE_BATCH) {
            for edge in batch.iter() {
                let head = (edge >> WEIGHT_BITS) as usize;
                _mm_prefetch::<_MM_HINT_T0>(self.raw.nodes.add(head).cast());
            }
            for edge in batch {
                let head = self.new_id((*edge >> WEIGHT_BITS) as usize);
                *edge = head << WEIGHT_BITS | (*edge & WEIGHT_MASK);
            }
        }
    }

    /// Gives back this worker's stretch of a region nobody needs any more; all workers doing
    /// it together beat one worker doing it all while the rest wait.
    unsafe fn release(&self, region: *mut u8, bytes: usize, worker: usize) {
        let stretch = bytes.div_ceil(self.threads).next_multiple_of(4096);
        let from = (worker * stretch).min(bytes);
        madvise(region.add(from), (bytes - from).min(stretch), MADV_DONTNEED);
    }

    unsafe fn work(&self, worker: usize) {
        let ids = self.forward.node_count;
        self.parse();
        self.next_step(|| {});
        self.each_job(self.rows.claimed.len(), |shard| {
            self.raw.build_shard(&self.rows, shard)
        });
        self.next_step(|| {});
        self.grow_cells(worker);
        self.next_step(|| drop(self.layout.set(self.plan_layout())));
        let layout = self.layout.get().unwrap();
        self.release(self.raw.edges.cast(), self.raw.edge_bytes, worker);
        let chunks = self.chunks.lock().unwrap().clone();
        self.each_job(chunks.len(), |job| self.rewrite_forward(chunks[job]));
        self.next_step(|| {});
        self.release(self.raw.nodes.cast(), self.nodes * 4, worker);
        // Pull the weight of each edge's arc back into its slot; arcs with no arc back are set aside.
        let mut extra = Extras::new();
        self.each_job(ids.div_ceil(CELL), |cell| {
            for x in cell * CELL..((cell + 1) * CELL).min(ids) {
                let entry = *self.forward.nodes.add(x);
                let start = ((entry << 1) >> (START_SHIFT + 1)) as usize;
                for (i, &edge) in self.forward.edges_at(entry).iter().enumerate() {
                    let y = (edge >> WEIGHT_BITS) as usize;
                    let mut back = NO_REV;
                    for &f in self.forward.edges_at(*self.forward.nodes.add(y)) {
                        if (f >> WEIGHT_BITS) as usize == x {
                            back = back.min((f & WEIGHT_MASK) as u16);
                        }
                    }
                    *self.rev.add(start + i) = back;
                    if back == NO_REV {
                        let mut w = edge & WEIGHT_MASK;
                        if w == WEIGHT_MASK {
                            w = layout.heavy[&(x as u32, y as u32)];
                        }
                        extra.push((y as u32, x as u32, w));
                    }
                }
            }
        });
        self.extras_raw.lock().unwrap().append(&mut extra);
        self.next_step(|| {
            let mut all = std::mem::take(&mut *self.extras_raw.lock().unwrap());
            all.sort_unstable_by_key(|e| e.0);
            for e in &all {
                *self.forward.nodes.add(e.0 as usize) |= FLAG;
            }
            drop(self.extras.set(all));
        });
        let extras = self.extras.get().unwrap();

        let mut forward = Side::new(&self.forward, &layout.heavy, self.rev, extras);
        let mut backward = Side::new(&self.forward, &layout.heavy, self.rev, extras);
        self.each_job(layout.queries.len(), |index| {
            let (s, t) = layout.queries[index];
            let distance = answer(s as usize, t as usize, &mut forward, &mut backward);
            self.answers.lock().unwrap()[index] = distance;
        });
    }
}

/// A monotone bucket queue. A light edge reaches less than WINDOW ahead, so every live key
/// sits in a circular window of exact distances, and popping is finding the next set bit,
/// with none of a heap's unpredictable comparisons. Heavy edges wait in `far`.
#[derive(Default)]
struct Queue {
    heads: Vec<u32>,        // item index + 1, per bucket
    items: Vec<(u32, u32)>, // node, and the next item in its bucket or in the free list
    free: u32,              // item index + 1, of the most recently popped item
    far: BinaryHeap<Reverse<(u64, u32)>>,
    now: u64,
    live: usize,
}

impl Queue {
    #[inline(always)]
    fn push(&mut self, distance: u64, node: usize) {
        self.live += 1;
        if distance - self.now >= WINDOW as u64 {
            self.far.push(Reverse((distance, node as u32)));
            return;
        }
        let bucket = distance as usize & (WINDOW - 1);
        let item = (node as u32, self.heads[bucket]);
        // Popped items are reused, so memory follows the queue's size, not the pushes.
        if self.free == 0 {
            self.items.push(item);
            self.heads[bucket] = self.items.len() as u32;
        } else {
            let slot = self.free as usize - 1;
            self.free = self.items[slot].1;
            self.items[slot] = item;
            self.heads[bucket] = slot as u32 + 1;
        }
    }

    /// Moves `now` to the smallest key, and says whether there is one.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        if !self.far.is_empty() {
            self.bring_far_keys_in();
        }
        // Every key in the window is below every key still in `far`, so the first one is it.
        for steps in 0..WINDOW as u64 {
            if self.heads[(self.now + steps) as usize & (WINDOW - 1)] != 0 {
                self.now += steps;
                return true;
            }
        }
        // The window is empty: jump to the nearest far key.
        let Some(Reverse((distance, node))) = self.far.pop() else {
            return false;
        };
        self.now = distance;
        self.live -= 1;
        self.push(distance, node as usize);
        true
    }

    #[cold]
    fn bring_far_keys_in(&mut self) {
        while let Some(&Reverse((distance, node))) = self.far.peek() {
            if distance - self.now >= WINDOW as u64 {
                return;
            }
            self.far.pop();
            self.live -= 1;
            self.push(distance, node as usize);
        }
    }
}

/// One direction of a search. Distances live in blocks that are handed out when a search
/// first reaches a stretch of the numbering, so a search costs what it touches, and cell
/// order keeps that small.
struct Side<'a> {
    graph: &'a Graph,
    heavy: &'a Heavy, // the weights that do not fit in an edge
    rev: *const u16,  // see Run::rev
    extras: &'a [(u32, u32, u32)],
    blocks: Vec<u32>, // block index + 1, per stretch of nodes
    distances: Vec<u16>,
    wide: HashMap<usize, u64>, // the distances that are too big
    touched: Vec<u32>,
    queue: Queue,
}

impl<'a> Side<'a> {
    fn new(
        graph: &'a Graph,
        heavy: &'a Heavy,
        rev: *const u16,
        extras: &'a [(u32, u32, u32)],
    ) -> Side<'a> {
        Side {
            graph,
            heavy,
            rev,
            extras,
            blocks: vec![0; (graph.node_count >> BLOCK_SHIFT) + 1],
            distances: Vec::new(),
            wide: HashMap::new(),
            touched: Vec::new(),
            queue: Queue {
                heads: vec![0; WINDOW],
                ..Default::default()
            },
        }
    }

    /// Forgets the search, at a cost of what it touched.
    fn finish(&mut self) {
        for &stretch in &self.touched {
            self.blocks[stretch as usize] = 0;
        }
        self.touched.clear();
        self.distances.clear();
        // A long search's blocks go back to the system, so that memory follows the searches
        // under way, not the longest one each thread has run.
        if self.distances.capacity() > 1 << 16 {
            self.distances = Vec::new();
        }
        self.wide.clear();
        let queue = &mut self.queue;
        queue.heads.fill(0);
        (queue.items.clear(), queue.far.clear());
        (queue.free, queue.now, queue.live) = (0, 0, 0);
    }

    /// Where a node's distance is kept, if its block exists.
    #[inline(always)]
    unsafe fn slot(&self, node: usize) -> Option<usize> {
        match *self.blocks.get_unchecked(node >> BLOCK_SHIFT) {
            0 => None,
            block => Some((block as usize - 1) * BLOCK_NODES + (node & (BLOCK_NODES - 1))),
        }
    }

    #[inline(always)]
    unsafe fn get(&self, node: usize) -> u64 {
        let Some(at) = self.slot(node) else {
            return UNREACHED;
        };
        match *self.distances.get_unchecked(at) {
            NONE => self.wide.get(&node).copied().unwrap_or(UNREACHED),
            distance => distance as u64,
        }
    }

    #[inline(always)]
    unsafe fn relax(&mut self, node: usize, distance: u64) {
        let at = self.slot(node).unwrap_or_else(|| self.new_block(node));
        let known = *self.distances.get_unchecked(at);
        // Only the mark is not below every distance that fits, so only it needs a look.
        if distance < known as u64
            || (known == NONE && self.wide.get(&node).map_or(true, |&k| distance < k))
        {
            if distance >= NONE as u64 {
                self.wide.insert(node, distance);
            }
            *self.distances.get_unchecked_mut(at) = distance.min(NONE as u64) as u16;
            self.queue.push(distance, node);
            // The edges of a new label are read soon, and the graph is too big for the cache.
            let start = (*self.graph.nodes.add(node) << 1) >> (START_SHIFT + 1);
            _mm_prefetch::<_MM_HINT_T1>(self.graph.edges.add(start as usize).cast());
        }
    }

    /// Follows an edge of the node being settled. A label that reaches the limit of
    /// `best - other.queue.now` cannot be part of a better path unless the other side has
    /// already settled `head`, so it is not queued and the other side is asked instead.
    #[inline(always)]
    unsafe fn reach(&mut self, head: usize, weight: u64, other: &Side, best: &mut u64) {
        let distance = self.queue.now + weight;
        if distance < best.saturating_sub(other.queue.now) {
            self.relax(head, distance);
        } else {
            *best = (*best).min(distance.saturating_add(other.get(head)));
        }
    }

    /// Settles the nearest node of this side: it may complete a path with the other side, and
    /// its edges give labels to its neighbours.
    #[inline(always)]
    unsafe fn settle<const BACK: bool>(&mut self, other: &Side, best: &mut u64) {
        // Take a node from the bucket `advance` found.
        let queue = &mut self.queue;
        let bucket = queue.now as usize & (WINDOW - 1);
        let item = queue.heads[bucket] as usize - 1;
        let (node, next) = queue.items[item];
        queue.items[item].1 = queue.free;
        queue.free = item as u32 + 1;
        queue.heads[bucket] = next;
        queue.live -= 1;
        let (node, distance) = (node as usize, queue.now);
        if distance > self.get(node) {
            return;
        }
        *best = (*best).min(distance.saturating_add(other.get(node)));
        let graph = self.graph;
        let entry = *graph.nodes.add(node);
        if !BACK {
            for &edge in graph.edges_at(entry) {
                let head = (edge >> WEIGHT_BITS) as usize;
                let mut weight = edge & WEIGHT_MASK;
                if weight == WEIGHT_MASK {
                    weight = self.heavy[&(node as u32, head as u32)];
                }
                self.reach(head, weight as u64, other, best);
            }
            return;
        }
        let start = ((entry << 1) >> (START_SHIFT + 1)) as usize;
        for (i, &edge) in graph.edges_at(entry).iter().enumerate() {
            let head = (edge >> WEIGHT_BITS) as usize;
            let back = *self.rev.add(start + i);
            if back == NO_REV {
                continue;
            }
            let mut weight = back as u32;
            if weight == WEIGHT_MASK {
                weight = self.heavy[&(head as u32, node as u32)];
            }
            self.reach(head, weight as u64, other, best);
        }
        if entry & FLAG != 0 {
            let list = self.extras;
            let from = list.partition_point(|e| (e.0 as usize) < node);
            for &(_, tail, weight) in list[from..].iter().take_while(|e| e.0 as usize == node) {
                self.reach(tail as usize, weight as u64, other, best);
            }
        }
    }

    /// Hands out the next block, to the stretch of nodes `node` is in, and returns where its
    /// distance is kept.
    #[cold]
    fn new_block(&mut self, node: usize) -> usize {
        let start = self.distances.len();
        self.distances.resize(start + BLOCK_NODES, NONE);
        let stretch = node >> BLOCK_SHIFT;
        self.blocks[stretch] = (start / BLOCK_NODES) as u32 + 1;
        self.touched.push(stretch as u32);
        start + (node & (BLOCK_NODES - 1))
    }
}

/// The shortest distance from `s` to `t`, or UNREACHED. The two sides take turns, the one with
/// fewer live labels first.
unsafe fn answer(s: usize, t: usize, forward: &mut Side, backward: &mut Side) -> u64 {
    forward.relax(s, 0);
    backward.relax(t, 0);
    let mut best = UNREACHED;
    // When a side runs dry it has settled everything it can reach, so any path was seen.
    while forward.queue.advance() && backward.queue.advance() {
        if forward.queue.now + backward.queue.now >= best {
            break;
        }
        if forward.queue.live <= backward.queue.live {
            forward.settle::<false>(backward, &mut best);
        } else {
            backward.settle::<true>(forward, &mut best);
        }
    }
    forward.finish();
    backward.finish();
    best
}

/// The unsigned decimal numbers in `text`, whatever separates them.
fn numbers(text: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let text = std::str::from_utf8(text).unwrap();
    text.split_ascii_whitespace()
        .map(|word| word.parse().unwrap())
}

fn main() {
    // Huge pages would round the per-thread buffers up to 2 MiB each. And a fixed malloc
    // threshold (-3 is M_MMAP_THRESHOLD; glibc would otherwise raise it after the first big
    // free), so that big blocks always come from and go back to mmap.
    unsafe { (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0), mallopt(-3, 1 << 17)) };
    let file = File::open("/data/graph.txt").unwrap();
    let text_len = file.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    let text = std::fs::read("/data/queries.txt").unwrap();
    let ids: Vec<_> = numbers(&text).map(|id| id as u32).collect();
    let raw_queries: Vec<_> = ids.chunks(2).map(|pair| (pair[0], pair[1])).collect();
    let run = unsafe {
        let text = map_file(&file, text_len);
        let mut header = numbers(&text[..64.min(text_len)]);
        let (nodes, edges) = (header.next().unwrap(), header.next().unwrap());
        let shards = (nodes >> SHARD_SHIFT) + 1;
        // Each worker's last window of new ids may be partly unused.
        let ids = nodes + threads * CELL;
        Run {
            text,
            nodes,
            threads,
            next_job: AtomicUsize::new(0),
            step: Barrier::new(threads),
            rows: Rows::new(shards),
            raw: Graph::new(shards << SHARD_SHIFT, edges),
            // Room for the unused tails of every worker's EDGE_CLAIM spans.
            forward: Graph::new(ids, edges + (edges >> 3) + (1 << 20)),
            rev: reserve((edges + (edges >> 3) + (1 << 20)) * 2 + 4096).cast(),
            extras_raw: Mutex::new(Vec::new()),
            extras: OnceLock::new(),
            heavy: Mutex::new(Vec::new()),
            next_id: AtomicU32::new(0),
            chunks: Mutex::new(Vec::new()),
            layout: OnceLock::new(),
            answers: Mutex::new(vec![0; raw_queries.len()]),
            raw_queries,
        }
    };
    thread::scope(|s| {
        for worker in 1..threads {
            let run = &run;
            s.spawn(move || unsafe { run.work(worker) });
        }
        unsafe { run.work(0) };
    });
    // No path is u64::MAX, which prints as -1 once read as signed. Real distances are below 2^63.
    let answers = run.answers.into_inner().unwrap();
    let out: String = (answers.iter())
        .map(|&d| format!("{}\n", d as i64))
        .collect();
    std::fs::write("/out/distances.txt", out).unwrap();
}

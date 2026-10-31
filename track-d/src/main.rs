//! Speed Demon 2026, Track D (Traverse).
//!
//! Node ids are a random relabelling of a road map, so in file order every step of a search
//! is a cache miss. Most of the program therefore exists to undo that:
//!
//! 1. Parse edges into rows, sharded by tail node, which needs no shared counters.
//! 2. Build an adjacency array from the shards, in any node order.
//! 3. Grow compact cells of nodes over that graph and renumber in cell order, so that nodes
//!    close on the map are close in memory.
//! 4. Rewrite the forward and backward adjacency arrays in the new order.
//! 5. Answer each query with a bidirectional Dijkstra, on one thread per query.
//!
//! An edge is 4 bytes: 21 bits of head and 11 of weight. Only about 3,500 of the 5,000,000
//! weights need more, so those edges live in a side list and their tails carry a flag.
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::ptr::read_unaligned;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex, OnceLock};
use std::thread;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x4000;
const MADV_DONTNEED: i32 = 4;
const MADV_HUGEPAGE: i32 = 14;
const MADV_NOHUGEPAGE: i32 = 15;
const MADV_POPULATE_READ: i32 = 22;

const CLAIM: usize = 1 << 20;
const SHARD_SHIFT: u32 = 13; // few enough shards that a worker's write positions stay in L1
const SHARD_NODES: usize = 1 << SHARD_SHIFT;
const SHARD_ROWS: usize = SHARD_NODES * 16; // every node has fewer than 16 light edges
const CHUNK_ROWS: usize = 64;
const WEIGHT_BITS: u32 = 11;
const LIGHT: u64 = 1 << WEIGHT_BITS; // weights below this fit in an edge
const DEGREE_MASK: u32 = 15;
const HEAVY_FLAG: u32 = 16;
const START_SHIFT: u32 = 5;
const CELL: usize = 1 << 10;
const SEEDS_KEPT: usize = 32; // of a finished cell's queue
const FAR_AHEAD: usize = 12; // queue entries ahead to prefetch a node
const NEAR_AHEAD: usize = 6; // queue entries ahead to prefetch its edges
const REWRITE_BATCH: usize = 32; // nodes translated together, after prefetching
const BLOCK_SHIFT: u32 = 6; // distances live in blocks of this many neighbouring nodes
const BLOCK_NODES: usize = 1 << BLOCK_SHIFT;
const WINDOW: usize = 1 << WEIGHT_BITS; // light edges reach at most this far ahead
const WINDOW_WORDS: usize = WINDOW / 64;
const LABEL_SHIFT: u32 = 26; // a claimed node is (worker + 1) << 26 | position in its order
const UNREACHED: u64 = u64::MAX;
const ZEROS: u64 = 0x3030_3030_3030_3030;

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
        let bytes = shards * SHARD_ROWS * size_of::<Row>();
        let rows = reserve(bytes, MADV_NOHUGEPAGE);
        Rows { rows: rows.cast(), claimed: (0..shards).map(|_| AtomicUsize::new(0)).collect() }
    }

    /// The rows of a shard, one chunk at a time, skipping each chunk's unused end.
    unsafe fn each(&self, shard: usize, mut visit: impl FnMut(Row)) {
        let region = self.rows.add(shard * SHARD_ROWS);
        for offset in (0..self.claimed[shard].load(Relaxed)).step_by(CHUNK_ROWS) {
            for i in offset..offset + CHUNK_ROWS {
                let row = region.add(i).read_unaligned();
                if row.packed == 0 {
                    break;
                }
                visit(row);
            }
        }
    }

    unsafe fn release(&self) {
        let bytes = self.claimed.len() * SHARD_ROWS * size_of::<Row>();
        madvise(self.rows.cast(), bytes, MADV_DONTNEED);
    }
}

/// One worker's write position in each shard.
struct Writer {
    cursors: Vec<(*mut Row, *mut Row)>,
}

impl Writer {
    fn new(shards: usize) -> Writer {
        let empty = std::ptr::null_mut();
        Writer { cursors: vec![(empty, empty); shards] }
    }

    #[inline(always)]
    unsafe fn push(&mut self, rows: &Rows, node: usize, other: usize, weight: u64) {
        let shard = node >> SHARD_SHIFT;
        let cursor = self.cursors.get_unchecked_mut(shard);
        if cursor.0 == cursor.1 {
            let offset = rows.claimed.get_unchecked(shard).fetch_add(CHUNK_ROWS, Relaxed);
            cursor.0 = rows.rows.add(shard * SHARD_ROWS + offset);
            cursor.1 = cursor.0.add(CHUNK_ROWS);
        }
        let packed = ((other as u32 + 1) << WEIGHT_BITS) | weight as u32;
        cursor.0.write_unaligned(Row { packed, low: (node & (SHARD_NODES - 1)) as u16 });
        cursor.0 = cursor.0.add(1);
    }
}

/// An adjacency array. Each node's entry is start << 5 | heavy flag | degree, so a node's
/// edges take one load to find, and shards may sit in any order.
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
            nodes: reserve(node_count * 4, MADV_HUGEPAGE).cast(),
            edges: reserve(edge_bytes, MADV_HUGEPAGE).cast(),
            node_count,
            edge_bytes,
            next_edge: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    unsafe fn edges_of(&self, node: usize) -> &[u32] {
        let entry = *self.nodes.add(node);
        let start = (entry >> START_SHIFT) as usize;
        std::slice::from_raw_parts(self.edges.add(start), (entry & DEGREE_MASK) as usize)
    }

    unsafe fn release(&self) {
        madvise(self.nodes.cast(), self.node_count * 4, MADV_DONTNEED);
        madvise(self.edges.cast(), self.edge_bytes, MADV_DONTNEED);
    }

    /// Fills in one shard of nodes from its rows.
    unsafe fn build_shard(&self, rows: &Rows, shard: usize) {
        let mut degree = [0u32; SHARD_NODES];
        rows.each(shard, |row| degree[row.low as usize] += 1);
        let total: u32 = degree.iter().sum();
        let base = self.next_edge.fetch_add(total as usize, Relaxed);
        let mut at = [0u32; SHARD_NODES];
        let mut next = base as u32;
        for low in 0..SHARD_NODES {
            at[low] = next;
            assert!(degree[low] <= DEGREE_MASK);
            let node = (shard << SHARD_SHIFT) + low;
            if node < self.node_count {
                *self.nodes.add(node) = next << START_SHIFT | degree[low];
            }
            next += degree[low];
        }
        rows.each(shard, |row| {
            let slot = &mut at[row.low as usize];
            *self.edges.add(*slot as usize) = row.packed - (1 << WEIGHT_BITS);
            *slot += 1;
        });
    }
}

/// Anonymous memory that costs nothing until it is touched.
unsafe fn reserve(bytes: usize, advice: i32) -> *mut u8 {
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    let region = mmap(std::ptr::null_mut(), bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    madvise(region, bytes, advice);
    region
}

unsafe fn map_file(file: &File, len: usize) -> *const u8 {
    let span = len.div_ceil(4096) * 4096 + 4096;
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    let base = mmap(std::ptr::null_mut(), span, PROT_READ, flags, -1, 0);
    mmap(base, len, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
    base
}

unsafe fn read(p: *const u8) -> u64 {
    read_unaligned(p.cast())
}

/// The number in the `len` bytes before `end`. Up to eight digits take the same few
/// instructions whatever the length, so varying lengths do not mispredict.
#[inline(always)]
unsafe fn number_before(end: *const u8, len: usize) -> u64 {
    if len > 8 {
        return std::slice::from_raw_parts(end.sub(len), len)
            .iter()
            .fold(0, |value, &digit| value * 10 + (digit - b'0') as u64);
    }
    let v = (read(end.sub(8)) ^ ZEROS) >> (64 - 8 * len) << (64 - 8 * len);
    let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

/// A node whose weight does not fit in an edge, with the edge's other end and its weight.
type Heavy = (u32, u32, u32);

struct Layout {
    node_base: Vec<u32>, // the first new id of each worker's nodes
    edge_base: Vec<usize>,
    queries: Vec<(u32, u32)>,
    heavy_out: Vec<Heavy>,
    heavy_in: Vec<Heavy>,
}

struct Run {
    text: *const u8,
    text_len: usize,
    nodes: usize,
    shards: usize,
    threads: usize,
    next_claim: AtomicUsize,
    next_job: AtomicUsize,
    step: Barrier,
    rows: Rows,
    raw: Graph,
    forward: Graph,
    backward: Graph,
    labels: *mut AtomicU32,
    heavy: Mutex<Vec<Heavy>>,
    claimed: Mutex<Vec<(usize, usize, usize)>>, // worker, nodes, edges
    layout: OnceLock<Layout>,
    raw_queries: Vec<(u32, u32)>,
    answers: Vec<AtomicU64>,
}
unsafe impl Sync for Run {}

impl Run {
    unsafe fn parse(&self, writer: &mut Writer) {
        let text = self.text;
        let mapped_end = self.text_len.div_ceil(4096) * 4096;
        let digit_floor = _mm256_set1_epi8(b'0' as i8);
        let mut heavy = Vec::new();
        loop {
            let claim = self.next_claim.fetch_add(CLAIM, Relaxed);
            if claim >= self.text_len {
                break;
            }
            madvise(text.add(claim) as *mut u8, CLAIM.min(mapped_end - claim), MADV_POPULATE_READ);
            // A line belongs to the claim it starts in. The header belongs to nobody.
            let mut at = claim;
            if claim == 0 || *text.add(claim - 1) != b'\n' {
                while *text.add(at) != b'\n' {
                    at += 1;
                }
                at += 1;
            }
            let end = (claim + CLAIM).min(self.text_len);
            while at < end {
                // A line is at most 27 bytes, so one load holds all three of its separators.
                let line = text.add(at);
                let chunk = _mm256_loadu_si256(line.cast());
                let mut separators = _mm256_movemask_epi8(_mm256_cmpgt_epi8(digit_floor, chunk)) as u32;
                let first = separators.trailing_zeros() as usize;
                separators &= separators - 1;
                let second = separators.trailing_zeros() as usize;
                separators &= separators - 1;
                let third = separators.trailing_zeros() as usize;
                let u = number_before(line.add(first), first) as usize;
                let v = number_before(line.add(second), second - first - 1) as usize;
                let w = number_before(line.add(third), third - second - 1);
                at += third + 1;
                if w < LIGHT {
                    writer.push(&self.rows, u, v, w);
                } else {
                    heavy.push((u as u32, v as u32, w as u32));
                }
            }
        }
        self.heavy.lock().unwrap().extend(heavy);
    }

    /// Claims jobs 0..count, one at a time, until none are left.
    fn each_job(&self, count: usize, mut job: impl FnMut(usize)) {
        loop {
            let next = self.next_job.fetch_add(1, Relaxed);
            if next >= count {
                return;
            }
            job(next);
        }
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
    /// With random ids a node's entry, its edges and its label are three cache misses. The
    /// queue says what comes next, so each is prefetched in turn before it is needed.
    unsafe fn grow_cells(&self, worker: usize) -> (Vec<u32>, usize) {
        let mut order: Vec<u32> = Vec::with_capacity(2 * self.nodes / self.threads);
        let mut seeds: Vec<u32> = Vec::new();
        let mut queue: Vec<u32> = Vec::new();
        let mut scan = 0..0;
        let mut edges = 0;
        loop {
            let seed = match seeds.pop() {
                Some(seed) => seed as usize,
                None => match scan.next() {
                    Some(node) => node,
                    None => {
                        let start = self.next_job.fetch_add(4096, Relaxed);
                        if start >= self.nodes {
                            break;
                        }
                        scan = start..(start + 4096).min(self.nodes);
                        continue;
                    }
                },
            };
            if (*self.labels.add(seed)).load(Relaxed) != 0 {
                continue;
            }
            queue.clear();
            queue.push(seed as u32);
            let (mut head, mut size) = (0, 0);
            while head < queue.len() && size < CELL {
                if let Some(&ahead) = queue.get(head + FAR_AHEAD) {
                    _mm_prefetch::<_MM_HINT_T0>(self.raw.nodes.add(ahead as usize).cast());
                    _mm_prefetch::<_MM_HINT_T0>(self.labels.add(ahead as usize).cast());
                }
                if let Some(&ahead) = queue.get(head + NEAR_AHEAD) {
                    let start = *self.raw.nodes.add(ahead as usize) >> START_SHIFT;
                    _mm_prefetch::<_MM_HINT_T0>(self.raw.edges.add(start as usize).cast());
                }
                let node = queue[head] as usize;
                head += 1;
                let label = &*self.labels.add(node);
                let value = ((worker as u32 + 1) << LABEL_SHIFT) | order.len() as u32;
                if label.load(Relaxed) != 0 || label.compare_exchange(0, value, Relaxed, Relaxed).is_err() {
                    continue;
                }
                order.push(node as u32);
                size += 1;
                let neighbours = self.raw.edges_of(node);
                edges += neighbours.len();
                queue.extend(neighbours.iter().map(|&edge| edge >> WEIGHT_BITS));
            }
            let kept = queue.len().saturating_sub(SEEDS_KEPT).max(head);
            seeds.extend_from_slice(&queue[kept..]);
        }
        (order, edges)
    }

    #[inline(always)]
    unsafe fn new_id(&self, base: &[u32], old: usize) -> u32 {
        let label = (*self.labels.add(old)).load(Relaxed);
        let worker = (label >> LABEL_SHIFT) as usize - 1;
        *base.get_unchecked(worker) + (label & ((1 << LABEL_SHIFT) - 1))
    }

    fn plan_layout(&self) -> Layout {
        let mut claimed = self.claimed.lock().unwrap();
        claimed.sort_unstable();
        let (mut nodes, mut edges) = (0, 0);
        let (mut node_base, mut edge_base) = (Vec::new(), Vec::new());
        for &(_, node_count, edge_count) in claimed.iter() {
            node_base.push(nodes as u32);
            edge_base.push(edges);
            nodes += node_count;
            edges += edge_count;
        }
        let map = |old: u32| unsafe { self.new_id(&node_base, old as usize) };
        let queries = self.raw_queries.iter().map(|&(s, t)| (map(s), map(t))).collect();
        let heavy = self.heavy.lock().unwrap();
        let mut heavy_out: Vec<Heavy> = heavy.iter().map(|&(u, v, w)| (map(u), map(v), w)).collect();
        let mut heavy_in: Vec<Heavy> = heavy_out.iter().map(|&(u, v, w)| (v, u, w)).collect();
        heavy_out.sort_unstable();
        heavy_in.sort_unstable();
        Layout { node_base, edge_base, queries, heavy_out, heavy_in }
    }

    /// Translates every edge head to its new id, which is a random lookup per edge. Working
    /// in small batches lets each one prefetch all it will touch before it touches it.
    unsafe fn rewrite_forward(&self, worker: usize, order: &[u32], layout: &Layout) {
        let mut id = layout.node_base[worker] as usize;
        let mut at = layout.edge_base[worker];
        for batch in order.chunks(REWRITE_BATCH) {
            for &old in batch {
                _mm_prefetch::<_MM_HINT_T0>(self.raw.nodes.add(old as usize).cast());
            }
            for &old in batch {
                let start = *self.raw.nodes.add(old as usize) >> START_SHIFT;
                _mm_prefetch::<_MM_HINT_T0>(self.raw.edges.add(start as usize).cast());
            }
            for &old in batch {
                for &edge in self.raw.edges_of(old as usize) {
                    let head = (edge >> WEIGHT_BITS) as usize;
                    _mm_prefetch::<_MM_HINT_T0>(self.labels.add(head).cast());
                }
            }
            for &old in batch {
                let edges = self.raw.edges_of(old as usize);
                *self.forward.nodes.add(id) = (at as u32) << START_SHIFT | edges.len() as u32;
                id += 1;
                for &edge in edges {
                    let head = self.new_id(&layout.node_base, (edge >> WEIGHT_BITS) as usize);
                    *self.forward.edges.add(at) = head << WEIGHT_BITS | (edge & (LIGHT as u32 - 1));
                    at += 1;
                    (*self.backward.nodes.add(head as usize).cast::<AtomicU32>()).fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Turns the in-degree counts into entries that point just past each node's edges.
    unsafe fn lay_out_backward(&self) {
        let mut end = 0u32;
        for node in 0..self.nodes {
            let entry = self.backward.nodes.add(node);
            assert!(*entry <= DEGREE_MASK);
            end += *entry;
            *entry |= end << START_SHIFT;
        }
    }

    /// Each edge takes the slot below its head's end, and the entry ends up at its start.
    unsafe fn fill_backward(&self, nodes: std::ops::Range<usize>) {
        for node in nodes {
            for &edge in self.forward.edges_of(node) {
                let head = (edge >> WEIGHT_BITS) as usize;
                let entry = &*self.backward.nodes.add(head).cast::<AtomicU32>();
                let slot = (entry.fetch_sub(1 << START_SHIFT, Relaxed) >> START_SHIFT) - 1;
                let weight = edge & (LIGHT as u32 - 1);
                *self.backward.edges.add(slot as usize) = (node as u32) << WEIGHT_BITS | weight;
            }
        }
    }

    unsafe fn mark_heavy(&self, graph: &Graph, heavy: &[Heavy]) {
        for &(node, _, _) in heavy {
            *graph.nodes.add(node as usize) |= HEAVY_FLAG;
        }
    }

    unsafe fn work(&self, worker: usize) {
        let mut writer = Writer::new(self.shards);
        self.parse(&mut writer);
        self.next_step(|| {});
        self.each_job(self.shards, |shard| self.raw.build_shard(&self.rows, shard));
        self.next_step(|| self.rows.release());
        let (order, edges) = self.grow_cells(worker);
        self.claimed.lock().unwrap().push((worker, order.len(), edges));
        self.next_step(|| {
            let _ = self.layout.set(self.plan_layout());
        });
        let layout = self.layout.get().unwrap();
        self.rewrite_forward(worker, &order, layout);
        let first_id = layout.node_base[worker] as usize;
        let own_nodes = first_id..first_id + order.len();
        drop(order);
        self.next_step(|| {
            self.raw.release();
            madvise(self.labels.cast(), self.nodes * 4, MADV_DONTNEED);
            self.lay_out_backward();
        });
        self.fill_backward(own_nodes);
        self.next_step(|| {
            self.mark_heavy(&self.forward, &layout.heavy_out);
            self.mark_heavy(&self.backward, &layout.heavy_in);
        });

        let mut forward = Side::new(&self.forward);
        let mut backward = Side::new(&self.backward);
        self.each_job(layout.queries.len(), |index| {
            let (s, t) = layout.queries[index];
            let distance = self.answer(s as usize, t as usize, layout, &mut forward, &mut backward);
            self.answers[index].store(distance, Relaxed);
        });
    }

    unsafe fn answer(
        &self,
        s: usize,
        t: usize,
        layout: &Layout,
        forward: &mut Side,
        backward: &mut Side,
    ) -> u64 {
        if s == t {
            return 0;
        }
        forward.relax(s, 0);
        backward.relax(t, 0);
        let mut best = UNREACHED;
        // When a side runs dry it has settled everything it can reach, so any path was seen.
        while forward.queue.advance() && backward.queue.advance() {
            if forward.queue.now + backward.queue.now >= best {
                break;
            }
            let forward_turn = forward.queue.live <= backward.queue.live;
            let (side, other, graph, heavy) = if forward_turn {
                (&mut *forward, &mut *backward, &self.forward, &layout.heavy_out)
            } else {
                (&mut *backward, &mut *forward, &self.backward, &layout.heavy_in)
            };
            let distance = side.queue.now;
            let node = side.queue.pop();
            if distance > side.get(node) {
                continue;
            }
            let met = other.get(node);
            if met != UNREACHED {
                best = best.min(distance + met);
            }
            // A label at or past this limit cannot be part of a better path unless the
            // other side has already settled that node, which `reach` checks instead.
            let limit = best.saturating_sub(other.queue.now);
            for &edge in graph.edges_of(node) {
                let weight = (edge & (LIGHT as u32 - 1)) as u64;
                side.reach((edge >> WEIGHT_BITS) as usize, distance + weight, limit, other, &mut best);
            }
            if *graph.nodes.add(node) & HEAVY_FLAG != 0 {
                let from = heavy.partition_point(|h| (h.0 as usize) < node);
                for &(_, head, weight) in heavy[from..].iter().take_while(|h| h.0 as usize == node) {
                    side.reach(head as usize, distance + weight as u64, limit, other, &mut best);
                }
            }
        }
        forward.finish();
        backward.finish();
        best
    }
}

/// A monotone bucket queue. A light edge reaches less than WINDOW ahead, so every live key
/// sits in a circular window of exact distances, and popping is finding the next set bit,
/// with none of a heap's unpredictable comparisons. Heavy edges wait in `far`.
struct Queue {
    heads: Vec<u32>, // item index + 1, per bucket
    occupied: [u64; WINDOW_WORDS],
    items: Vec<(u32, u32)>, // node, and the next item in its bucket
    far: BinaryHeap<Reverse<(u64, u32)>>,
    now: u64,
    live: usize,
}

impl Queue {
    fn new() -> Queue {
        Queue {
            heads: vec![0; WINDOW],
            occupied: [0; WINDOW_WORDS],
            items: Vec::new(),
            far: BinaryHeap::new(),
            now: 0,
            live: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, distance: u64, node: usize) {
        self.live += 1;
        if distance - self.now >= WINDOW as u64 {
            self.far.push(Reverse((distance, node as u32)));
            return;
        }
        let bucket = distance as usize & (WINDOW - 1);
        self.items.push((node as u32, self.heads[bucket]));
        self.heads[bucket] = self.items.len() as u32;
        self.occupied[bucket / 64] |= 1 << (bucket % 64);
    }

    /// Moves `now` to the smallest key, and says whether there is one.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        loop {
            if !self.far.is_empty() {
                self.bring_far_keys_in();
            }
            let from = self.now as usize & (WINDOW - 1);
            let word = from / 64;
            let here = self.occupied[word] >> (from % 64);
            if here != 0 {
                self.now += here.trailing_zeros() as u64;
                return true;
            }
            if let Some(steps) = self.next_bucket(from) {
                self.now += steps as u64;
                continue;
            }
            match self.far.pop() {
                Some(Reverse((distance, node))) => {
                    self.now = distance;
                    self.live -= 1;
                    self.push(distance, node as usize);
                }
                None => return false,
            }
        }
    }

    /// How far ahead of `from` the next occupied bucket lies, looking past its own word.
    #[inline(never)]
    fn next_bucket(&self, from: usize) -> Option<usize> {
        let word = from / 64;
        for step in 1..=WINDOW_WORDS {
            let index = (word + step) % WINDOW_WORDS;
            let mut bits = self.occupied[index];
            if step == WINDOW_WORDS {
                bits &= (1 << (from % 64)) - 1; // back at the start, below `from` only
            }
            if bits != 0 {
                let bucket = index * 64 + bits.trailing_zeros() as usize;
                return Some((bucket + WINDOW - from) % WINDOW);
            }
        }
        None
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

    /// Takes a node from the bucket `advance` found.
    #[inline(always)]
    fn pop(&mut self) -> usize {
        let bucket = self.now as usize & (WINDOW - 1);
        let item = self.heads[bucket] as usize - 1;
        let (node, next) = self.items[item];
        self.heads[bucket] = next;
        if next == 0 {
            self.occupied[bucket / 64] &= !(1 << (bucket % 64));
        }
        self.live -= 1;
        node as usize
    }

    fn clear(&mut self) {
        for word in 0..WINDOW_WORDS {
            while self.occupied[word] != 0 {
                let bit = self.occupied[word].trailing_zeros() as usize;
                self.heads[word * 64 + bit] = 0;
                self.occupied[word] &= self.occupied[word] - 1;
            }
        }
        self.items.clear();
        self.far.clear();
        self.now = 0;
        self.live = 0;
    }
}

/// One direction of a search. Distances live in blocks that are handed out when a search
/// first reaches a stretch of the numbering, so a search costs what it touches, and cell
/// order keeps that small.
struct Side {
    blocks: Vec<u32>, // block index + 1, per stretch of nodes
    distances: Vec<u64>,
    used: usize,
    touched: Vec<u32>,
    queue: Queue,
    graph: *const u32,
    edges: *const u32,
}

impl Side {
    fn new(graph: &Graph) -> Side {
        Side {
            blocks: vec![0; (graph.node_count >> BLOCK_SHIFT) + 1],
            distances: Vec::new(),
            used: 0,
            touched: Vec::new(),
            queue: Queue::new(),
            graph: graph.nodes,
            edges: graph.edges,
        }
    }

    fn finish(&mut self) {
        for &stretch in &self.touched {
            self.blocks[stretch as usize] = 0;
        }
        self.touched.clear();
        self.used = 0;
        self.queue.clear();
    }

    #[inline(always)]
    fn get(&self, node: usize) -> u64 {
        match unsafe { *self.blocks.get_unchecked(node >> BLOCK_SHIFT) } {
            0 => UNREACHED,
            block => unsafe {
                let at = (block as usize - 1) * BLOCK_NODES + (node & (BLOCK_NODES - 1));
                *self.distances.get_unchecked(at)
            },
        }
    }

    #[inline(always)]
    fn relax(&mut self, node: usize, distance: u64) {
        let stretch = node >> BLOCK_SHIFT;
        let mut block = unsafe { *self.blocks.get_unchecked(stretch) };
        if block == 0 {
            block = self.new_block(stretch);
        }
        let at = (block as usize - 1) * BLOCK_NODES + (node & (BLOCK_NODES - 1));
        let known = unsafe { self.distances.get_unchecked_mut(at) };
        if distance < *known {
            *known = distance;
            self.queue.push(distance, node);
            // The edges of a new label are read soon, and the graph is too big for the cache.
            unsafe {
                let start = *self.graph.add(node) >> START_SHIFT;
                _mm_prefetch::<_MM_HINT_T1>(self.edges.add(start as usize).cast());
            }
        }
    }

    #[inline(always)]
    fn reach(&mut self, node: usize, distance: u64, limit: u64, other: &Side, best: &mut u64) {
        if distance < limit {
            self.relax(node, distance);
        } else {
            let met = other.get(node);
            if met != UNREACHED {
                *best = (*best).min(distance + met);
            }
        }
    }

    #[cold]
    fn new_block(&mut self, stretch: usize) -> u32 {
        let start = self.used * BLOCK_NODES;
        if self.distances.len() < start + BLOCK_NODES {
            self.distances.resize(start + BLOCK_NODES, UNREACHED);
        }
        self.distances[start..start + BLOCK_NODES].fill(UNREACHED);
        self.used += 1;
        self.blocks[stretch] = self.used as u32;
        self.touched.push(stretch as u32);
        self.used as u32
    }
}

fn read_queries() -> Vec<(u32, u32)> {
    let text = std::fs::read("/data/queries.txt").unwrap();
    let number = |word: &[u8]| word.iter().fold(0, |value, &digit| value * 10 + (digit - b'0') as u32);
    text.split(|&c| c == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            let split = line.iter().position(|&c| c == b' ').unwrap();
            (number(&line[..split]), number(&line[split + 1..]))
        })
        .collect()
}

fn main() {
    let file = File::open("/data/graph.txt").unwrap();
    let text_len = file.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    let raw_queries = read_queries();
    let run = unsafe {
        let text = map_file(&file, text_len);
        let header = std::slice::from_raw_parts(text, 64);
        let line_end = header.iter().position(|&c| c == b'\n').unwrap();
        let mut words = header[..line_end].split(|&c| c == b' ');
        let mut number = || words.next().unwrap().iter().fold(0, |v, &d| v * 10 + (d - b'0') as usize);
        let (nodes, edges) = (number(), number());
        let shards = (nodes >> SHARD_SHIFT) + 1;
        Run {
            text,
            text_len,
            nodes,
            shards,
            threads,
            next_claim: AtomicUsize::new(0),
            next_job: AtomicUsize::new(0),
            step: Barrier::new(threads),
            rows: Rows::new(shards),
            raw: Graph::new(nodes, edges),
            forward: Graph::new(nodes, edges),
            backward: Graph::new(nodes, edges),
            labels: reserve(nodes * 4, MADV_HUGEPAGE).cast(),
            heavy: Mutex::new(Vec::new()),
            claimed: Mutex::new(Vec::new()),
            layout: OnceLock::new(),
            answers: raw_queries.iter().map(|_| AtomicU64::new(UNREACHED)).collect(),
            raw_queries,
        }
    };
    thread::scope(|s| {
        for worker in 1..threads {
            let builder = thread::Builder::new().stack_size(1 << 19);
            let run = &run;
            builder.spawn_scoped(s, move || unsafe { run.work(worker) }).unwrap();
        }
        unsafe { run.work(0) };
    });

    let mut out = String::with_capacity(run.answers.len() * 12);
    for answer in &run.answers {
        match answer.load(Relaxed) {
            UNREACHED => out.push_str("-1\n"),
            distance => out.push_str(&format!("{distance}\n")),
        }
    }
    File::create("/out/distances.txt").unwrap().write_all(out.as_bytes()).unwrap();
}

//! Speed Demon 2026, Track B (Index).
//!
//! The queries are readable from the start, so this answers those 50,000 queries instead of
//! indexing a corpus:
//!
//! 1. Read the queries and put every term in one table. A token that is not in it is
//!    dropped after one hash, whatever else the corpus holds.
//! 2. Index at document level only. Terms common enough to deserve it get a bitmap, chosen
//!    from a small sample of the corpus. The rest become lists of document ids, written as
//!    rows that need no shared counters. Rows last one phase of 50 MiB of text, then become
//!    16-bit postings, so that rows and postings never fill memory together.
//! 3. Store no positions. A phrase is checked during the scan, when the document's tokens
//!    are at hand: when its rarest term appears, the neighbouring tokens' hashes are compared
//!    with the phrase. A match writes a row for a pseudo-term that stands for the phrase, so
//!    a phrase query is then a one-term lookup, and no document is ever read twice.
//! 4. Evaluate BOOL strictly left to right on whichever of list or bitmap suits each step.
//!    Queries are answered once per phase, on that phase's postings and bitmaps, into a
//!    running tally per query (count, id sum, ten smallest ids), so no full index ever exists.
//!
//! Threads. One worker per CPU, started as a tree (`spawn_tree`), all running `Run::work`:
//! parse a share of `queries.txt` and intern its terms into the shared bucket table, sample
//! 12 claims, plan anchors and bitmaps, then for each phase scan its 256 KiB claims, build
//! its postings shard by shard (zeroing a slice of the idle bitmaps), and answer it. The
//! answers of phase k - 1 run on whichever threads have run out of scan claims in phase k,
//! so no thread waits for the slowest scanner. Work is shared through atomic job counters,
//! one set per phase, so no counter is ever reset. `main` writes the answer parts in file
//! order.
//!
//! The term table is 2^15 buckets of one cache line, each with 7 tags and 8 actions
//! (`Bucket`), on a huge page. A token's action says what the scan does with it: nothing,
//! set a bit in a bitmap, write a row for a list, or also start a phrase check.
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};
use std::{arch::x86_64::*, fs::File, io::Write, os::fd::AsRawFd, ptr::null_mut, thread};

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, arg: usize, a: usize, b: usize, c: usize) -> i32;
    fn syscall(number: i64, ...) -> i64;
    fn memchr(text: *const u8, byte: i32, len: usize) -> *const u8;
}
// Protection 3 is PROT_READ | PROT_WRITE, and flags 0x12 are MAP_PRIVATE | MAP_FIXED, a file
// over pages already reserved.
const ANONYMOUS: i32 = 0x4022; // MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE

const HUGE_PAGE: usize = 1 << 21;
const CLAIM: usize = 1 << 18;
// 50 MiB of text between turning rows into postings.
const PHASE_CLAIMS: usize = 200;
// u16s before a list: a sentinel while it fills, so duplicates drop with no branch.
const HEADER: usize = 1;
// Slices each of the phrase plan and the bucket plan is cut into.
const JOBS: usize = 16;
const TALLY_IDS: usize = 10;
const BATCH: usize = 1 << 14; // bytes of text between the token pass and the term pass
const BUCKETS: usize = 1 << 15;
const BUCKET_SLOTS: usize = 7;
const ID_BLOCK: usize = 32; // term ids a thread takes at once while parsing queries
const MAX_TERMS: usize = BUCKETS * BUCKET_SLOTS + 64 * ID_BLOCK;
const AHEAD: usize = 20; // tokens between a bucket's prefetch and its lookup
const HEAVY: u32 = 1 << 31; // an action: 0 for no term, a bitmap index here, or a term id + 1
const OVERFLOW: u32 = 1 << 30; // in place of 0: the bucket is full, so look further
const ANCHOR: u32 = 1 << 29; // added to a term's action: some phrase starts its check here
const MAX_HEAVY: usize = 4096;
const SHARDS: usize = 64;
const DOC_BITS: u32 = 18;
const CHUNK: usize = 64; // rows; the first of each chunk names the claim that wrote it
const QUERY_BATCH: usize = 64;
// A worker's share of the one reservation for batch space.
const WORKSPACE: usize = 32 << 20;
// Words in a phrase, at most (the grammar allows 2 to 8).
const MAX_WORDS: usize = 8;

/// One cache line. The last action answers for a hash found in no slot, so that a miss
/// takes the same path as a hit.
#[repr(C, align(64))]
struct Bucket {
    tags: [u32; BUCKET_SLOTS + 1],
    actions: [u32; BUCKET_SLOTS + 1],
}

/// A worker's space for a batch of text.
struct Buffers {
    starts: &'static mut [u32],
    ends: &'static mut [u32],
    lines: &'static mut [u32],
    hashes: &'static mut [u64],
    heavy_hits: &'static mut [u32], // the bitmap index of each heavy token, with the HEAVY flag
    heavy_ends: &'static mut [u32], // per line, how many heavy hits end with it
    group: &'static mut [u64],      // the heavy terms of 64 documents, one row of bits each
    list_hits: &'static mut [u64],  // action << 32 | document
    anchor_hits: &'static mut [u64], // term + 1 << 40 | document << 20 | token (the shift drops the action's flags)
    counts: &'static mut [u32],      // in the sample, how often each term occurs
}

struct Term {
    first: u32,   // the header slot of the term's list in this phase, or its bitmap index
    count: u32,   // documents containing the term
    phrases: u32, // the first phrase anchored here, + 1
    heavy: bool,
    in_bool: bool,
}

/// A phrase, the hashes of its words, and the pseudo-term holding its matches.
struct Phrase {
    term: u32,
    len: u32,
    pivot: u32, // which word is the anchor, chosen from the sample
    next: u32,  // the next phrase with the same anchor, + 1
    words: [u64; MAX_WORDS],
}

/// Queries, flattened: each is a run of (op, term) with the first op unused. A phrase is one
/// step, on its pseudo-term.
#[derive(Clone, Copy, PartialEq)]
enum Op {
    And,
    Or,
    Not,
}

/// The current result of a BOOL query, either in the index or in scratch space.
#[derive(Clone, Copy)]
enum Set {
    List(&'static [u16]),
    Bits(&'static [u64]),
}

#[derive(Default)]
struct Scratch {
    lists: [Vec<u16>; 2],
    bits: Vec<u64>,
    marks: Vec<u64>, // all zeros between uses
}

/// What the phases so far found for one query. All zeros is the empty answer, so the array
/// of them can come straight from fresh pages.
struct Tally {
    count: u32,
    sum: u64,
    first: [u32; TALLY_IDS], // the smallest ids, filled from the first phases on
}

impl Tally {
    /// Adds a phase's result as a list of ids relative to the phase's first document.
    fn add_list(&mut self, base: u32, list: &[u16]) {
        let sum = list.iter().map(|&doc| doc as u64).sum();
        let ids = list.iter().map(|&doc| doc as u32);
        self.add(base, list.len() as u64, sum, ids);
    }

    /// Adds a phase's `count` ids, `sum` of them, relative to the phase's first document, with
    /// `ids` the smallest first: they fill the tally's slots if it has room for more.
    fn add(&mut self, base: u32, count: u64, sum: u64, ids: impl Iterator<Item = u32>) {
        let room = (self.count as usize).min(TALLY_IDS);
        for (slot, id) in self.first[room..].iter_mut().zip(ids) {
            *slot = base + id;
        }
        self.count += count as u32;
        self.sum += sum + base as u64 * count;
    }

    /// Adds a phase's result as a bitmap, whose length is a multiple of 4 words. The sum of the
    /// ids in a byte is 8 * its index * its popcount, plus the sum of its bit positions, and
    /// both come from nibble lookups, 32 bytes at a time.
    fn add_bits(&mut self, base: u32, bits: &[u64]) {
        let (count, sum) = unsafe {
            let table =
                |row: [i8; 16]| _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast()));
            let pick = |table, index| _mm256_shuffle_epi8(table, index);
            let ones = table([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]);
            let places = table([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6]);
            let weights = _mm256_set1_epi64x(0x3830_2820_1810_0800); // 8 * byte index in a word
            let (low, zero) = (_mm256_set1_epi8(15), _mm256_setzero_si256());
            let (mut counts, mut sums, mut offsets, mut starts) = (zero, zero, zero, zero);
            let mut first = _mm256_setr_epi64x(0, 64, 128, 192); // the ids of each word's first bit
            for block in bits.chunks_exact(4) {
                let word = _mm256_loadu_si256(block.as_ptr().cast());
                let lo = _mm256_and_si256(word, low);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(word), low);
                let high_ones = pick(ones, hi);
                let per_byte = _mm256_add_epi8(pick(ones, lo), high_ones);
                let low_places = _mm256_add_epi8(pick(places, lo), pick(places, hi));
                // The high nibble's positions are 4 more than in the table.
                let positions = _mm256_add_epi8(low_places, _mm256_slli_epi16::<2>(high_ones));
                let per_word = _mm256_sad_epu8(per_byte, zero);
                counts = _mm256_add_epi64(counts, per_word);
                sums = _mm256_add_epi64(sums, _mm256_sad_epu8(positions, zero));
                starts = _mm256_add_epi64(starts, _mm256_mul_epu32(per_word, first));
                let weighted = _mm256_maddubs_epi16(per_byte, weights);
                offsets =
                    _mm256_add_epi32(offsets, _mm256_madd_epi16(weighted, _mm256_set1_epi16(1)));
                first = _mm256_add_epi64(first, _mm256_set1_epi64x(256));
            }
            // Every lane is below 2^32, so the halves of the 64-bit ones can be added as 32-bit.
            let total = |v: __m256i| {
                let lanes: [u32; 8] = std::mem::transmute(v);
                lanes.iter().map(|&lane| lane as u64).sum::<u64>()
            };
            (total(counts), total(sums) + total(starts) + total(offsets))
        };
        let ids = bits
            .iter()
            .enumerate()
            .filter(|(_, &word)| word != 0)
            .flat_map(|(index, &word)| {
                (0..64)
                    .filter(move |bit| word >> bit & 1 == 1)
                    .map(move |bit| index as u32 * 64 + bit)
            });
        self.add(base, count, sum, ids);
    }

    fn write(&self, out: &mut Vec<u8>) {
        put_number(out, self.count as u64);
        put_number(out, self.sum);
        for &doc in &self.first[..(self.count as usize).min(TALLY_IDS)] {
            put_number(out, doc as u64);
        }
        *out.last_mut().unwrap() = b'\n'; // in place of the last space
    }
}

struct Run {
    text: &'static [u8],
    docs: usize,
    words_per_bitmap: usize,
    buckets: *mut Bucket,
    terms: *mut Term,
    term_count: AtomicUsize,
    queries: &'static [u8],
    shares: Vec<OnceLock<Parsed>>, // one per thread, set when it has parsed its part of the file
    phrases: *mut Phrase,
    sample: &'static [AtomicU32], // per term, the documents of the sample that hold it
    // Counters of the steps before the first phase: sample claims taken, phrase plans taken,
    // bucket plans taken, bitmaps given out, documents in the sample, and phrases parsed.
    setup: [AtomicUsize; 6],
    // Per phase, so that no phase needs resetting: the claims taken, the jobs that build taken,
    // the jobs that answer taken, and the next free posting.
    phase: Vec<[AtomicUsize; 4]>,
    phase_base: Vec<u32>, // the first document of each phase, then the document count
    // The barrier: how many threads have arrived, and the round they wait to see change.
    arrived: AtomicU32,
    round: AtomicU32,
    bitmaps: *mut u64,
    rows: *mut u32,
    claimed: Vec<AtomicUsize>,
    postings: *mut u16,
    parts: Vec<Mutex<Vec<u8>>>, // the output of each batch of queries
    workspace: *mut u8,
}
unsafe impl Sync for Run {}

/// Room for `count` values on a 2 MiB boundary, with huge pages allowed or not (`madvise`
/// advice 14 is MADV_HUGEPAGE, 15 is MADV_NOHUGEPAGE).
unsafe fn reserve<T>(count: usize, huge: bool) -> *mut T {
    let bytes = count * size_of::<T>();
    let region = mmap(null_mut(), bytes + HUGE_PAGE, 3, ANONYMOUS, -1, 0);
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, bytes, 15 - huge as i32);
    aligned.cast()
}

/// Gives the pages back to the kernel, which reads them as zeros again.
unsafe fn discard(at: *const u8, len: usize) {
    madvise(at as *mut u8, len, 4);
}

/// Lazily backed space on small pages, so that large bounds cost only what is touched.
unsafe fn space<T>(count: usize) -> &'static mut [T] {
    std::slice::from_raw_parts_mut(reserve(count, false), count)
}

/// Hands out the parts of one worker's share of a reservation made before the threads start.
/// One `mmap` for all the workers, instead of eight each: every `mmap` and `madvise` takes the
/// address-space lock for writing, which stalls the page faults of the threads already running.
unsafe fn take<T>(next: &mut *mut u8, count: usize) -> &'static mut [T] {
    let at = *next;
    *next = at.add((count * size_of::<T>()).next_multiple_of(4096));
    std::slice::from_raw_parts_mut(at.cast(), count)
}

unsafe fn map_file(path: &str) -> &'static [u8] {
    let file = File::open(path).unwrap();
    let len = file.metadata().unwrap().len() as usize;
    // The file goes over the start of a longer reservation, so that a vector load past its end
    // reads zeros. An empty file fails to map and leaves just the zeros.
    let span = len.div_ceil(4096) * 4096 + 4096;
    let base = mmap(null_mut(), span, 1, ANONYMOUS, -1, 0);
    mmap(base, len, 1, 0x12, file.as_raw_fd(), 0);
    std::slice::from_raw_parts(base, len)
}

/// The hash of a token in any case, never 0, which marks an unused slot. Tokens are compared
/// by this hash alone, and the table keeps 47 bits of it (15 pick the bucket, 32 are the
/// tag), so a token that is no query term matches one with odds of about 3 in 10,000 over the
/// whole corpus: a deliberate shortcut, for one load and compare per token. Every token byte
/// is a letter or digit, and `| 0x20` lowers letters and leaves digits, so folding needs no
/// check on each byte.
const KEY: i64 = 0x9E37_79B9_7F4A_7C15_u64 as i64;

/// A load from `MASKS` at `16 - len` is ones for the first `len` bytes and zeros after.
static MASKS: [[u8; 16]; 2] = [[255; 16], [0; 16]];

/// The hashes of two tokens at once, one in each half of a vector.
#[inline(always)]
unsafe fn hash_pair(a: *const u8, len_a: usize, b: *const u8, len_b: usize) -> (u64, u64) {
    let load = |at: *const u8| _mm_loadu_si128(at.cast());
    let mask_of = |len: usize| load(MASKS.as_flattened().as_ptr().add(16 - len));
    let pair = _mm256_set_m128i(load(b), load(a));
    let mask = _mm256_set_m128i(mask_of(len_b), mask_of(len_a));
    let key = _mm256_and_si256(_mm256_or_si256(pair, _mm256_set1_epi8(0x20)), mask);
    // Two AES rounds spread every byte over the whole block, in fewer uops than a multiply.
    let mixed = _mm256_aesenc_epi128(_mm256_aesenc_epi128(key, _mm256_set1_epi64x(KEY)), key);
    let folded = _mm256_xor_si256(mixed, _mm256_srli_si256::<8>(mixed));
    let [low, high] = [
        _mm256_extract_epi64::<0>(folded),
        _mm256_extract_epi64::<2>(folded),
    ];
    (low as u64 | 1, high as u64 | 1)
}

#[inline(always)]
unsafe fn hash_of(token: *const u8, len: usize) -> u64 {
    if len > 16 {
        long_hash(token, len)
    } else {
        hash_pair(token, len, token, len).0
    }
}

/// A token's action, with no branch on whether a query uses it.
#[inline(always)]
unsafe fn action_in(buckets: *const Bucket, hash: u64) -> u32 {
    let bucket = buckets.add(bucket_of(hash));
    let wanted = _mm256_set1_epi32(hash as i32);
    let same = _mm256_cmpeq_epi32(_mm256_load_si256(bucket.cast()), wanted);
    let found = _mm256_movemask_ps(_mm256_castsi256_ps(same)) as u32 | 1 << BUCKET_SLOTS;
    let action = (*bucket).actions[found.trailing_zeros() as usize];
    if action == OVERFLOW {
        return Run::action_past(buckets, hash);
    }
    action
}

#[inline(always)]
fn bucket_of(hash: u64) -> usize {
    (hash >> (64 - BUCKETS.ilog2())) as usize
}

/// A token over 16 bytes (the longest in the corpus has 17) is its first 16 bytes hashed and
/// mixed with the hash of the rest.
#[cold]
unsafe fn long_hash(token: *const u8, len: usize) -> u64 {
    (hash_of(token, 16).rotate_left(29) ^ hash_of(token.add(16), len - 16)) | 1
}

impl Run {
    /// The action of a token whose bucket is full, so that its term may be in a later one.
    #[cold]
    unsafe fn action_past(buckets: *const Bucket, hash: u64) -> u32 {
        let mut index = bucket_of(hash);
        loop {
            let bucket = &*buckets.add(index);
            match bucket.tags.iter().position(|&tag| tag == hash as u32) {
                Some(slot) => return bucket.actions[slot],
                None if bucket.actions[BUCKET_SLOTS] == 0 => return 0,
                None => index = (index + 1) % BUCKETS,
            }
        }
    }

    #[cold]
    unsafe fn claim_chunk(&self, shard: usize) -> *mut u32 {
        let offset = self.claimed[shard].fetch_add(CHUNK, Relaxed);
        self.rows.add(shard * self.stride() + offset)
    }

    /// Words between two shards' row regions: a whole number of cache lines.
    fn stride(&self) -> usize {
        (self.text.len() + 4096 + 15) & !15
    }

    /// Reads one claim. With SAMPLE set, it only counts how often each term appears.
    unsafe fn index_claim<const SAMPLE: bool>(&self, claim: usize, b: &mut Buffers) -> usize {
        let (start, mut doc) = aligned_line(self.text, self.docs, claim * CLAIM);
        let end = aligned_line(self.text, self.docs, claim * CLAIM + CLAIM).0;
        if start >= end {
            return 0;
        }
        let (first_doc, text) = (doc, self.text.as_ptr());

        // Write combining: rows of a shard gather in one cache line here and leave it as one
        // 64-byte store, `fill` words in, bound for `dest` in the chunk that ends at `end`.
        let mut lines = [Line([0; 16]); SHARDS];
        let mut cursors = [(null_mut::<u32>(), null_mut::<u32>(), 0usize); SHARDS];
        let (mut token_count, mut end_count, mut line_count) = (0, 0, 0);
        // `carry` is whether the previous 64 bytes ended inside a token.
        let (mut at, mut carry, mut first_token) = (start, 0u64, true);

        let byte = |value: u8| _mm256_set1_epi8(value as i8);
        let (floor_lower, ceil_lower) = (byte(b'a' - 1), byte(b'z' + 1));
        let (floor_digit, ceil_digit) = (byte(b'0' - 1), byte(b'9' + 1));
        let (fold, newline) = (byte(0x20), byte(b'\n'));
        let between = |x, floor, ceil| {
            _mm256_and_si256(_mm256_cmpgt_epi8(x, floor), _mm256_cmpgt_epi8(ceil, x))
        };

        while at < end {
            let batch_end = (at + BATCH).min(end);
            while at < batch_end {
                let classify = |at: usize| {
                    let chunk = _mm256_loadu_si256(text.add(at).cast());
                    let lower = _mm256_or_si256(chunk, fold);
                    let letters = between(lower, floor_lower, ceil_lower);
                    let digits = between(chunk, floor_digit, ceil_digit);
                    (
                        _mm256_movemask_epi8(_mm256_or_si256(letters, digits)) as u32 as u64,
                        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline)) as u32 as u64,
                    )
                };
                let (inside_low, breaks_low) = classify(at);
                let (inside_high, breaks_high) = classify(at + 32);
                let mut inside = inside_low | inside_high << 32;
                let mut breaks = breaks_low | breaks_high << 32;
                if at + 64 > end {
                    let keep = (1u64 << (end - at)).wrapping_sub(1);
                    inside &= keep;
                    breaks &= keep;
                }
                let shifted = inside << 1 | carry;
                carry = inside >> 63;
                let mut opened = inside & !shifted;
                let mut closed = !inside & shifted;
                // A line is recorded as the index of the token that follows it, the next id.
                while breaks != 0 {
                    let before = opened & ((1u64 << breaks.trailing_zeros()) - 1);
                    b.lines[line_count] = (token_count + before.count_ones() as usize) as u32;
                    line_count += 1;
                    breaks &= breaks - 1;
                }
                // Unconditional stores, so a varying token count needs no loop to predict. Twelve
                // cover almost every 64-byte step; the loops after take the rare rest.
                let (found, ended) = (opened.count_ones() as usize, closed.count_ones() as usize);
                let first_in = |bits: u64| (at + bits.trailing_zeros() as usize) as u32;
                for extra in 0..12 {
                    *b.starts.get_unchecked_mut(token_count + extra) = first_in(opened);
                    *b.ends.get_unchecked_mut(end_count + extra) = first_in(closed);
                    opened &= opened.wrapping_sub(1);
                    closed &= closed.wrapping_sub(1);
                }
                for extra in 12..found {
                    b.starts[token_count + extra] = first_in(opened);
                    opened &= opened - 1;
                }
                for extra in 12..ended {
                    b.ends[end_count + extra] = first_in(closed);
                    closed &= closed - 1;
                }
                token_count += found;
                end_count += ended;
                at += 64;
            }

            // Only whole documents go on, so that a phrase has all its tokens at hand.
            let ready = b.lines[line_count.saturating_sub(1)] as usize;
            let mut token = 0;
            if first_token && ready > 0 {
                first_token = false;
                token = 1; // the id of the claim's first document
                b.hashes[0] = 0;
            }
            let (starts, ends, hashes) =
                (b.starts.as_ptr(), b.ends.as_ptr(), b.hashes.as_mut_ptr());
            let span = |i: usize| (*starts.add(i) as usize, *ends.add(i) as usize);
            let mut i = token;
            while i + 1 < ready {
                let (from_a, to_a) = span(i);
                let (from_b, to_b) = span(i + 1);
                let (len_a, len_b) = (to_a - from_a, to_b - from_b);
                let (a, b) = (text.add(from_a), text.add(from_b));
                (*hashes.add(i), *hashes.add(i + 1)) = if (len_a | len_b) > 16 {
                    (hash_of(a, len_a), hash_of(b, len_b))
                } else {
                    hash_pair(a, len_a, b, len_b)
                };
                i += 2;
            }
            if i < ready {
                let (from, to) = span(i);
                *hashes.add(i) = hash_of(text.add(from), to - from);
            }
            let (mut heavy_count, mut list_count, mut anchor_count) = (0, 0, 0);
            let batch_doc = doc;
            // Local copies of the pointers: through `self` and `b` the compiler reloads them
            // after every store, as a store might have changed them.
            let buckets = self.buckets;
            let (heavy_hits, list_hits) = (b.heavy_hits.as_mut_ptr(), b.list_hits.as_mut_ptr());
            for line in 0..line_count {
                let to = b.lines[line] as usize;
                while token < to {
                    let ahead = bucket_of(*hashes.add(token + AHEAD));
                    _mm_prefetch::<_MM_HINT_T0>(buckets.add(ahead).cast());
                    let action = action_in(buckets, *hashes.add(token));
                    // All three stores happen, and only the right counts move.
                    *heavy_hits.add(heavy_count) = action;
                    heavy_count += (action >> 31) as usize;
                    *list_hits.add(list_count) = (action as u64) << 32 | doc as u64;
                    // A term id + 1, with or without ANCHOR: not 0 and not HEAVY.
                    list_count += (action.wrapping_sub(1) < HEAVY - 1) as usize;
                    // Rare enough to predict: about 1 token in 150 starts a phrase check.
                    if action & ANCHOR != 0 {
                        *b.anchor_hits.get_unchecked_mut(anchor_count) =
                            (action as u64) << 40 | (doc as u64) << 20 | token as u64;
                        anchor_count += 1;
                    }
                    token += 1;
                }
                *b.heavy_ends.get_unchecked_mut(line) = heavy_count as u32;
                // The token after a line is the next document's id. As hash 0 it also stops a
                // phrase check from running on into the next document. The last line stays
                // for the next batch.
                if line + 1 < line_count {
                    doc += 1;
                    b.hashes[to] = 0;
                    token = to + 1;
                }
            }
            for (i, &hit) in b.anchor_hits[..anchor_count].iter().enumerate() {
                let later = |ahead: usize| (b.anchor_hits[i + ahead] >> 40) as usize;
                _mm_prefetch::<_MM_HINT_T0>(self.terms.add(later(8).wrapping_sub(1)).cast());
                if let Some(term) = later(4).checked_sub(1) {
                    let first = (*self.terms.add(term)).phrases as usize;
                    _mm_prefetch::<_MM_HINT_T0>(self.phrases.add(first.wrapping_sub(1)).cast());
                }
                let token = hit as usize & ((1 << 20) - 1);
                let mut next = (*self.terms.add((hit >> 40) as usize - 1)).phrases;
                while next != 0 {
                    let phrase = &*self.phrases.add(next as usize - 1);
                    next = phrase.next;
                    let (len, pivot) = (phrase.len as usize, phrase.pivot as usize);
                    if token >= pivot
                        && token - pivot + len <= ready
                        && b.hashes[token - pivot..token - pivot + len] == phrase.words[..len]
                    {
                        b.list_hits[list_count] =
                            (phrase.term as u64 + 1) << 32 | (hit >> 20 & ((1 << 20) - 1));
                        list_count += 1;
                    }
                }
            }
            if !SAMPLE {
                // Each document's heavy terms are gathered as one row of bits, which stays in
                // cache; every 64 documents the rows are transposed into the bitmaps.
                let (words, mut from) = (self.heavy().div_ceil(64), 0);
                for line in 0..line_count {
                    let to = b.heavy_ends[line] as usize;
                    let rel = (batch_doc as usize + line
                        - self.phase_base[claim / PHASE_CLAIMS] as usize)
                        % 64;
                    let row = &mut b.group[rel * words..(rel + 1) * words];
                    for &action in &b.heavy_hits[from..to] {
                        let id = (action & !HEAVY) as usize;
                        row[id / 64] |= 1 << (id % 64);
                    }
                    from = to;
                    if rel == 63 {
                        self.flush_group(b.group, claim, batch_doc as usize + line);
                    }
                }
            }
            for &hit in &b.list_hits[..list_count] {
                let term = ((hit >> 32) as u32 & !ANCHOR) - 1;
                if SAMPLE {
                    b.counts[term as usize] += 1;
                    continue;
                }
                let shard = term as usize & (SHARDS - 1);
                let cursor = cursors.get_unchecked_mut(shard);
                let line = lines.get_unchecked_mut(shard);
                if cursor.0 == cursor.1 {
                    let at = self.claim_chunk(shard);
                    *cursor = (at, at.add(CHUNK), 1);
                    line.0[0] = claim as u32 + 1;
                }
                line.0[cursor.2] = ((term >> SHARDS.trailing_zeros()) + 1) << DOC_BITS | hit as u32;
                cursor.2 += 1;
                if cursor.2 == 16 {
                    store_line(cursor.0, line);
                    cursor.0 = cursor.0.add(16);
                    cursor.2 = 0;
                }
            }
            // A newline left over belongs to a document whose id comes in the next batch.
            b.lines[0] = 0;
            line_count = line_count.min(1);
            b.starts.copy_within(ready..token_count, 0);
            token_count -= ready;
            b.ends.copy_within(ready..end_count, 0);
            end_count -= ready;
        }
        if !SAMPLE {
            self.flush_group(b.group, claim, doc as usize);
            for (cursor, line) in cursors.iter().zip(lines.iter_mut()) {
                if cursor.2 > 0 {
                    line.0[cursor.2..].fill(0);
                    store_line(cursor.0, line);
                }
            }
            _mm_sfence();
        }
        // The documents this claim held, which the sample uses to scale its counts.
        doc - first_doc + 1
    }

    /// How many terms have a bitmap.
    fn heavy(&self) -> usize {
        self.setup[3].load(Relaxed).min(MAX_HEAVY)
    }

    /// ORs the group of 64 documents that holds `doc` into the phase's bitmaps and clears it.
    /// Row `j` has the heavy terms of the group's document `j`; transposed, a word is one
    /// term's bits for the group.
    unsafe fn flush_group(&self, group: &mut [u64], claim: usize, doc: usize) {
        let phase = claim / PHASE_CLAIMS;
        let rel = doc - self.phase_base[phase] as usize;
        let (heavy, out) = (self.heavy(), self.bitmap(0, phase).add(rel / 64));
        let words = heavy.div_ceil(64);
        for k in 0..words {
            let mut m: [u64; 64] =
                std::array::from_fn(|j| std::mem::take(&mut group[j * words + k]));
            for width in [32usize, 16, 8, 4, 2, 1] {
                // The bits of a word that sit in the low half of each block of `2 * width`.
                let mask = u64::MAX / ((1 << width) + 1);
                for r in (0..64).filter(|r| r & width == 0) {
                    let t = ((m[r] >> width) ^ m[r + width]) & mask;
                    m[r + width] ^= t;
                    m[r] ^= t << width;
                }
            }
            for (t, &bits) in m.iter().enumerate().take(heavy - k * 64) {
                *out.add((k * 64 + t) * self.words_per_bitmap) |= bits;
            }
        }
    }

    /// A barrier on one futex word. The standard one wakes its sleepers through a mutex, one
    /// after another, and threads were released up to 3 ms after the last arrival.
    fn barrier(&self) {
        let round = self.round.load(Acquire);
        if self.arrived.fetch_add(1, AcqRel) as usize + 1 == self.shares.len() {
            self.arrived.store(0, Relaxed);
            self.round.fetch_add(1, Release);
            // FUTEX_WAKE | FUTEX_PRIVATE_FLAG, every waiter.
            unsafe { syscall(202, self.round.as_ptr(), 129, i32::MAX) };
            return;
        }
        // Spinning first: a thread that sleeps wakes up to 3 ms late.
        for _ in 0..3000 {
            if self.round.load(Acquire) != round {
                return;
            }
            std::hint::spin_loop();
        }
        while self.round.load(Acquire) == round {
            // FUTEX_WAIT | FUTEX_PRIVATE_FLAG, while the round is still the one seen.
            unsafe { syscall(202, self.round.as_ptr(), 128, round, 0) };
        }
    }

    fn each_job(&self, counter: &AtomicUsize, count: usize, job: impl FnMut(usize)) {
        (0..)
            .map(|_| counter.fetch_add(1, Relaxed))
            .take_while(|&next| next < count)
            .for_each(job);
    }

    /// Uses the sample to give each phrase in a slice its rarest word as an anchor. Phrases
    /// anchored on one term are chained through the term by an atomic swap, in any order.
    unsafe fn plan_phrases(&self, slice: usize) {
        let count = self.setup[5].load(Relaxed);
        for index in count * slice / JOBS..count * (slice + 1) / JOBS {
            let phrase = &mut *self.phrases.add(index);
            // Every word is in the table, whose actions are still term ids + 1.
            let terms = phrase.words[..phrase.len as usize].iter();
            let terms = terms.map(|&hash| action_in(self.buckets, hash) as usize - 1);
            let (pivot, term) = (terms.enumerate())
                .min_by_key(|&(_, term)| self.sample[term].load(Relaxed))
                .unwrap();
            let anchor = &mut (*self.terms.add(term)).phrases;
            phrase.pivot = pivot as u32;
            phrase.next = AtomicU32::from_ptr(anchor).swap(index as u32 + 1, Relaxed);
        }
    }

    /// Gives the actions in a slice of the buckets their final form, once the phrases have been
    /// planned. Slices are taken by whichever thread is free.
    unsafe fn plan_buckets(&self, slice: usize) {
        let docs = self.setup[4].load(Relaxed);
        for index in BUCKETS * slice / JOBS..BUCKETS * (slice + 1) / JOBS {
            let bucket = &mut *self.buckets.add(index);
            for action in &mut bucket.actions[..BUCKET_SLOTS] {
                if *action == 0 {
                    continue;
                }
                let id = *action as usize - 1;
                let term = &mut *self.terms.add(id);
                if term.phrases != 0 {
                    *action |= ANCHOR;
                } else if !term.in_bool {
                    *action = 0;
                // A term in 1/32 of the sample's documents is cheaper as a bitmap.
                } else if self.sample[id].load(Relaxed) as usize * 32 >= docs {
                    let heavy = self.setup[3].fetch_add(1, Relaxed);
                    if heavy < MAX_HEAVY {
                        *action = HEAVY | heavy as u32;
                        (term.heavy, term.first) = (true, heavy as u32);
                    }
                }
            }
        }
    }

    /// Sorts a shard's rows into one list per term for this phase. Chunks are taken in claim
    /// order, and each claim writes in document order, so the result is sorted with no sorting
    /// of rows.
    unsafe fn build_shard(&self, shard: usize, phase: usize) {
        let base = self.phase_base[phase];
        let region = self.rows.add(shard * self.stride());
        let claimed = self.claimed[shard].load(Relaxed);
        let mut chunks: Vec<(u32, u32)> = (0..claimed)
            .step_by(CHUNK)
            .map(|at| (*region.add(at), at as u32))
            .collect();
        chunks.sort_unstable();
        let term_count = self.term_count.load(Relaxed);
        let terms = term_count.div_ceil(SHARDS) + 1;
        // Lists belong to one phase: last phase's are gone.
        for id in (shard..term_count).step_by(SHARDS) {
            (*self.terms.add(id)).count = 0;
        }
        let mut cursors = vec![0u32; terms];
        for_rows(region, &chunks, |low, _| {
            *cursors.get_unchecked_mut(low) += 1
        });
        // Until a list is complete its sentinel slot holds u16::MAX, so a document can always
        // be compared with the one before it. A term repeats within a document, but a
        // document's rows are adjacent.
        let lists = cursors.iter().filter(|&&size| size > 0).count();
        let total = cursors.iter().sum::<u32>() as usize + lists * HEADER;
        let mut at = self.phase[phase][3].fetch_add(total, Relaxed) as u32;
        for cursor in cursors.iter_mut().filter(|cursor| **cursor > 0) {
            let size = *cursor;
            *cursor = at + HEADER as u32;
            *self.postings.add(*cursor as usize - 1) = u16::MAX;
            at += size + HEADER as u32;
        }
        let firsts = cursors.clone();
        for_rows(region, &chunks, |low, doc| {
            let cursor = cursors.get_unchecked_mut(low);
            let slot = self.postings.add(*cursor as usize);
            *slot = (doc - base) as u16;
            *cursor += (*slot.sub(1) != *slot) as u32;
        });
        for low in (0..terms).filter(|&low| firsts[low] > 0) {
            let info = &mut *self.terms.add(low * SHARDS + shard);
            info.first = firsts[low] - HEADER as u32;
            info.count = cursors[low] - firsts[low];
        }
        std::ptr::write_bytes(region, 0, claimed);
        self.claimed[shard].store(0, Relaxed);
        // Zero this shard's share of the bitmap set that the next phase will fill.
        let heavy = self.heavy();
        let (from, to) = (heavy * shard / SHARDS, heavy * (shard + 1) / SHARDS);
        let first = self.bitmap(from as u32, phase + 1);
        std::ptr::write_bytes(first, 0, (to - from) * self.words_per_bitmap);
    }

    /// Bitmaps come in two sets, so that one is zeroed while the other is being answered from.
    /// A bit is a document's offset from the first document of the phase.
    unsafe fn bitmap(&self, index: u32, phase: usize) -> *mut u64 {
        let set = (phase & 1) * MAX_HEAVY + index as usize;
        self.bitmaps.add(set * self.words_per_bitmap)
    }

    /// A term's documents in this phase, as offsets from its first document.
    unsafe fn set_of(&self, term: u32, phase: usize) -> Set {
        let info = &*self.terms.add(term as usize);
        if info.heavy {
            let bits = self.bitmap(info.first, phase);
            return Set::Bits(std::slice::from_raw_parts(bits, self.words_per_bitmap));
        }
        let docs = self.postings.add(info.first as usize + HEADER);
        Set::List(std::slice::from_raw_parts(docs, info.count as usize))
    }

    unsafe fn answer_bool(&self, steps: &[(Op, u32)], phase: usize, scratch: &mut Scratch) -> Set {
        let mut current = self.set_of(steps[0].1, phase);
        for &(op, term) in &steps[1..] {
            // Nothing is left to AND with or to take away from, so skip the term's lookup.
            if op != Op::Or && matches!(current, Set::List([])) {
                continue;
            }
            current = apply(op, current, self.set_of(term, phase), scratch);
        }
        current
    }

    /// Adds this phase's share of each query in a batch to its tally. After the last phase the
    /// tallies are final, and the batch's lines are written.
    unsafe fn answer(&self, (share, first, batch): Unit, phase: usize, scratch: &mut Scratch) {
        let base = self.phase_base[phase];
        let end = (first + QUERY_BATCH).min(share.queries.len());
        let tallies = share.tallies.as_ptr() as *mut Tally;
        let steps_of = |at: usize| &share.steps[share.queries[at].clone()];
        // Term records and postings are scattered, so a query's records are requested two
        // queries before they are read and the start of its lists one query before.
        let prefetch = |at: usize, lists: bool| {
            for &(_, term) in (at < end).then(|| steps_of(at)).unwrap_or_default() {
                let info = &*self.terms.add(term as usize);
                if !lists {
                    _mm_prefetch::<_MM_HINT_T0>((info as *const Term).cast());
                } else if !info.heavy {
                    let list = self.postings.add(info.first as usize);
                    // At most 6 cache lines of the list.
                    for line in 0..(info.count as usize).div_ceil(32).min(6) {
                        _mm_prefetch::<_MM_HINT_T0>(list.add(line * 32).cast());
                    }
                }
            }
        };
        prefetch(first, false);
        prefetch(first + 1, false);
        for index in first..end {
            prefetch(index + 2, false);
            prefetch(index + 1, true);
            let steps = steps_of(index);
            let tally = &mut *tallies.add(index);
            match self.answer_bool(steps, phase, scratch) {
                Set::List(list) => tally.add_list(base, list),
                Set::Bits(bits) => tally.add_bits(base, bits),
            }
        }
        if phase + 1 == self.phase_base.len() - 1 {
            let mut out = Vec::with_capacity(QUERY_BATCH * 64);
            for index in first..end {
                (*tallies.add(index)).write(&mut out);
            }
            *self.parts[batch].lock().unwrap() = out;
        }
    }

    /// Hashes and interns the queries of this thread's share of the file, which starts and ends
    /// at lines: share `k` begins at the first line start at or after `k / threads`.
    unsafe fn parse_share(&self, k: usize) -> Parsed {
        let (size, shares) = (self.queries.len(), self.shares.len());
        let start = |k: usize| match k * size / shares {
            0 => 0,
            at => self.queries[at - 1..]
                .iter()
                .position(|&c| c == b'\n')
                .map_or(size, |n| at + n),
        };
        let (mut parsed, mut ids, mut op) = (Parsed::default(), 0..0, Op::And);
        for line in self.queries[start(k)..start(k + 1)].split(|&c| c == b'\n') {
            // A share ends with a newline, so its last piece is empty.
            if line.is_empty() {
                continue;
            }
            let phrase = line[0] == b'P';
            // Past `PHRASE "` and before the closing quote, or past `BOOL `.
            let body = &line[if phrase { 8 } else { 5 }..line.len() - phrase as usize];
            let (first, mut words, mut count) = (parsed.steps.len(), [0; MAX_WORDS], 0);
            for word in body.split(|&c| c == b' ') {
                match word {
                    b"AND" => op = Op::And,
                    b"OR" => op = Op::Or,
                    b"NOT" => op = Op::Not,
                    _ => {
                        let hash = hash_of(word.as_ptr(), word.len());
                        let term = intern(self, &mut ids, hash);
                        if phrase {
                            words[count] = hash;
                            count += 1;
                        } else {
                            (*self.terms.add(term as usize)).in_bool = true;
                            parsed.steps.push((op, term));
                        }
                    }
                }
            }
            if phrase {
                let term = new_id(self, &mut ids);
                parsed.steps.push((Op::And, term));
                // Fresh pages are zeros, which are the unplanned `pivot` and `next`.
                let slot = &mut *self.phrases.add(self.setup[5].fetch_add(1, Relaxed));
                (slot.term, slot.len, slot.words) = (term, count as u32, words);
            }
            parsed.queries.push(first..parsed.steps.len());
        }
        parsed.tallies = space(parsed.queries.len());
        parsed
    }

    unsafe fn work(&self, id: usize) {
        let tokens = (1 << 19) + 32; // of a batch and an unfinished document
        let mut next = self.workspace.add(id * WORKSPACE);
        let mut buffers = Buffers {
            starts: take(&mut next, tokens),
            ends: take(&mut next, tokens),
            lines: take(&mut next, BATCH / 2 + 64),
            hashes: take(&mut next, tokens + AHEAD),
            heavy_hits: take(&mut next, tokens),
            heavy_ends: take(&mut next, BATCH / 2 + 64),
            group: take(&mut next, 64 * MAX_HEAVY / 64),
            list_hits: take(&mut next, tokens),
            anchor_hits: take(&mut next, tokens),
            counts: take(&mut next, MAX_TERMS),
        };
        self.shares[id].set(self.parse_share(id)).ok();
        self.barrier();
        // Threads that wake late find fewer claims left, instead of delaying everyone.
        // 12 claims are read to choose the terms that get bitmaps.
        self.each_job(&self.setup[0], 12, |claim| {
            self.setup[4].fetch_add(self.index_claim::<true>(claim, &mut buffers), Relaxed);
        });
        let terms = self.term_count.load(Relaxed);
        for (total, &part) in self.sample.iter().zip(&buffers.counts[..terms]) {
            total.fetch_add(part, Relaxed);
        }
        // The counts are done with: give their pages back, so they do not add to the peak.
        discard(buffers.counts.as_ptr().cast(), MAX_TERMS * 4);
        self.barrier();
        self.each_job(&self.setup[1], JOBS, |slice| self.plan_phrases(slice));
        self.barrier();
        self.each_job(&self.setup[2], JOBS, |part| self.plan_buckets(part));
        self.barrier();

        // The batches of queries, in file order.
        let mut units = Vec::new();
        for share in self.shares.iter().map(|share| share.get().unwrap()) {
            for first in (0..share.queries.len()).step_by(QUERY_BATCH) {
                units.push((share, first, units.len()));
            }
        }
        let mut scratch = Scratch {
            bits: vec![0; self.words_per_bitmap],
            marks: vec![0; self.words_per_bitmap],
            ..Default::default()
        };
        let phases = self.phase_base.len() - 1;
        // Stage k scans phase k, and a thread that runs out of claims answers phase k - 1
        // while the others finish scanning: the postings and bitmaps of k - 1 are still
        // there, because the build of k comes after the barrier that ends every answer.
        for stage in 0..=phases {
            if stage < phases {
                // A claim past the end of the text finds no line to start on and does nothing.
                self.each_job(&self.phase[stage][0], PHASE_CLAIMS, |claim| {
                    self.index_claim::<false>(stage * PHASE_CLAIMS + claim, &mut buffers);
                });
            }
            if stage > 0 {
                self.each_job(&self.phase[stage - 1][2], units.len(), |batch| {
                    self.answer(units[batch], stage - 1, &mut scratch)
                });
            }
            if stage == phases {
                break;
            }
            self.barrier();
            if stage + 1 == phases {
                // The scan is over: drop the text from the page tables in one slice per thread,
                // instead of one thread tearing the whole mapping down at exit.
                let page =
                    |n: usize| (self.text.len() * n / self.shares.len()).div_ceil(4096) * 4096;
                discard(self.text.as_ptr().add(page(id)), page(id + 1) - page(id));
            }
            // Postings are u16 offsets from the phase's first document, with u16::MAX the sentinel.
            assert!(self.phase_base[stage + 1] - self.phase_base[stage] < u16::MAX as u32);
            self.each_job(&self.phase[stage][1], SHARDS, |shard| {
                self.build_shard(shard, stage)
            });
            self.barrier();
        }
    }
}

#[derive(Clone, Copy)]
#[repr(align(64))]
struct Line([u32; 16]);

/// One 64-byte store of a gathered line into its (aligned) place in a chunk.
#[inline(always)]
unsafe fn store_line(to: *mut u32, line: &Line) {
    let from = line.0.as_ptr().cast::<__m256i>();
    if STREAM {
        _mm256_stream_si256(to.cast(), _mm256_load_si256(from));
        _mm256_stream_si256(to.add(8).cast(), _mm256_load_si256(from.add(1)));
    } else {
        _mm256_store_si256(to.cast(), _mm256_load_si256(from));
        _mm256_store_si256(to.add(8).cast(), _mm256_load_si256(from.add(1)));
    }
}
const STREAM: bool = true;

/// Visits a shard's rows in document order.
#[inline(always)]
unsafe fn for_rows(region: *const u32, chunks: &[(u32, u32)], mut visit: impl FnMut(usize, u32)) {
    for &(_, at) in chunks {
        for i in 1..CHUNK {
            let row = *region.add(at as usize + i);
            if row == 0 {
                break;
            }
            visit((row >> DOC_BITS) as usize - 1, row & ((1 << DOC_BITS) - 1));
        }
    }
}

/// The document id a line starts with.
unsafe fn id_of(line: *const u8) -> usize {
    let digits = (0..).map(|i| *line.add(i)).take_while(|&c| c != b'\t');
    digits.fold(0, |id, digit| id * 10 + (digit - b'0') as usize)
}

/// The position and id of the first line of `text` at or after `at` whose id is a multiple of
/// 64, or the end of the text and the document count `docs` if there is none. Claims start on
/// such lines, so no two workers ever share a word of a bitmap.
unsafe fn aligned_line(text: &[u8], docs: usize, at: usize) -> (usize, usize) {
    let (base, len) = (text.as_ptr(), text.len());
    let past_line =
        |from: usize| memchr(base.add(from), b'\n' as i32, len - from) as usize - base as usize + 1;
    let mut line = at;
    if at > 0 && at < len && *base.add(at - 1) != b'\n' {
        line = past_line(at);
    }
    while line < len {
        let id = id_of(base.add(line));
        if id % 64 == 0 {
            return (line, id);
        }
        line = past_line(line);
    }
    (len, docs)
}

/// Appends the values of `a` that are in `b` (or, with `keep` false, are not). A list
/// much shorter than the other is searched for in it (`binary_search` takes the same number
/// of steps whatever the value, each one a conditional move, so there is no branch to
/// mispredict and the searches for neighbouring values overlap). Otherwise one list is marked in `marks`,
/// which is all zeros between calls, and the other is tested against the marks and the marks
/// cleared again. Both write every value and move the count only for a kept one, so neither
/// branches on the data.
unsafe fn filter_by(a: &[u16], b: &[u16], keep: bool, out: &mut Vec<u16>, marks: &mut [u64]) {
    out.reserve(a.len().max(b.len()));
    let target = out.as_mut_ptr();
    let mut found = 0;
    // A list 16 times shorter is searched for, not marked.
    if a.len() * 16 <= b.len() {
        for &doc in a {
            let present = b.binary_search(&doc).is_ok();
            *target.add(found) = doc;
            found += (present == keep) as usize;
        }
    } else {
        let (marked, tested) = if keep { (a, b) } else { (b, a) };
        for &doc in marked {
            *marks.get_unchecked_mut(doc as usize / 64) |= 1 << (doc % 64);
        }
        for &doc in tested {
            let present = *marks.get_unchecked(doc as usize / 64) >> (doc % 64) & 1;
            *target.add(found) = doc;
            found += (present == keep as u64) as usize;
        }
        for &doc in marked {
            *marks.get_unchecked_mut(doc as usize / 64) = 0;
        }
    }
    out.set_len(found);
}

fn union(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        out.push(x.min(y));
        i += (x <= y) as usize;
        j += (y <= x) as usize;
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// One step of a BOOL query. The result goes in the scratch list the current set is not in.
unsafe fn apply(op: Op, current: Set, term: Set, scratch: &mut Scratch) -> Set {
    // AND is symmetric: test a list against whatever it is combined with, the shorter list
    // when both are lists.
    let swap = match (op, current, term) {
        (Op::And, Set::Bits(_), Set::List(_)) => true,
        (Op::And, Set::List(a), Set::List(b)) => b.len() < a.len(),
        _ => false,
    };
    let (a, b) = [(current, term), (term, current)][swap as usize];
    scratch.lists.swap(0, 1); // the current set may be in the first
    let out = &mut scratch.lists[0];
    out.clear();
    let bits = &mut scratch.bits;
    let (words, keep, marks) = (bits.len(), op == Op::And, &mut scratch.marks);
    match (op, a, b) {
        (Op::And | Op::Not, Set::List(a), Set::List(b)) => filter_by(a, b, keep, out, marks),
        (Op::And | Op::Not, Set::List(a), Set::Bits(b)) => {
            // Keep what the bitmap holds for AND, and what it lacks for NOT.
            let held = |doc: u16| *b.get_unchecked(doc as usize / 64) >> (doc % 64) & 1 == 1;
            out.extend(a.iter().filter(|&&doc| held(doc) == keep));
        }
        // A merged list is cheaper than a bitmap while it is shorter than one.
        (Op::Or, Set::List(a), Set::List(b)) if a.len() + b.len() < words => union(a, b, out),
        _ => {
            // The result is a bitmap: the current set is put in one, then the term is folded in.
            let mark = |bits: &mut [u64], docs: &[u16], clear: bool| {
                for &doc in docs {
                    let word = bits.get_unchecked_mut(doc as usize / 64);
                    let bit = 1 << (doc % 64);
                    *word = if clear { *word & !bit } else { *word | bit };
                }
            };
            match a {
                Set::Bits(source) if source.as_ptr() != bits.as_ptr() => {
                    bits.copy_from_slice(source)
                }
                Set::Bits(_) => {}
                Set::List(docs) => {
                    bits.fill(0);
                    mark(bits, docs, false);
                }
            }
            match b {
                Set::List(docs) => mark(bits, docs, op == Op::Not),
                Set::Bits(other) => {
                    for (word, &other) in bits.iter_mut().zip(other) {
                        *word = match op {
                            Op::And => *word & other,
                            Op::Or => *word | other,
                            Op::Not => *word & !other,
                        };
                    }
                }
            }
            return Set::Bits(std::slice::from_raw_parts(bits.as_ptr(), words));
        }
    }
    Set::List(std::slice::from_raw_parts(out.as_ptr(), out.len()))
}

/// Appends a number and a space: the digits are written backwards, then reversed in place.
fn put_number(out: &mut Vec<u8>, mut value: u64) {
    let from = out.len();
    while out.len() == from || value > 0 {
        out.push(b'0' + (value % 10) as u8);
        value /= 10;
    }
    out[from..].reverse();
    out.push(b' ');
}

/// Adds a term to the table, or finds it, and returns its id. Every thread interns at once:
/// a slot is claimed by a compare-exchange on its tag, and whoever loses to the same tag waits
/// for the winner's id, which is stored right after.
unsafe fn intern(run: &Run, ids: &mut std::ops::Range<u32>, hash: u64) -> u32 {
    let mut index = bucket_of(hash);
    loop {
        let bucket = &mut *run.buckets.add(index);
        for slot in 0..BUCKET_SLOTS {
            let tag = AtomicU32::from_ptr(&mut bucket.tags[slot]);
            let action = AtomicU32::from_ptr(&mut bucket.actions[slot]);
            let mut seen = tag.load(Acquire);
            if seen == 0 {
                seen = match tag.compare_exchange(0, hash as u32, AcqRel, Acquire) {
                    Ok(_) => {
                        let term = new_id(run, ids);
                        action.store(term + 1, Release);
                        return term;
                    }
                    Err(other) => other,
                };
            }
            if seen == hash as u32 {
                while action.load(Acquire) == 0 {
                    std::hint::spin_loop();
                }
                return action.load(Acquire) - 1;
            }
        }
        AtomicU32::from_ptr(&mut bucket.actions[BUCKET_SLOTS]).store(OVERFLOW, Relaxed);
        index = (index + 1) % BUCKETS;
    }
}

/// A fresh term id. Ids come in blocks, one atomic add per block; an unused id is a term
/// nothing mentions.
unsafe fn new_id(run: &Run, ids: &mut std::ops::Range<u32>) -> u32 {
    ids.next().unwrap_or_else(|| {
        let first = run.term_count.fetch_add(ID_BLOCK, Relaxed) as u32;
        *ids = first + 1..first + ID_BLOCK as u32;
        first
    })
}

/// A batch of queries: its share, the first query in it, and its place in the output.
type Unit<'a> = (&'a Parsed, usize, usize);

/// One thread's share of the queries, with offsets that start at zero in each share.
#[derive(Default)]
struct Parsed {
    queries: Vec<std::ops::Range<usize>>, // each one's steps
    steps: Vec<(Op, u32)>,
    tallies: &'static mut [Tally],
}

fn spawn_tree<'scope, 'env>(scope: &'scope thread::Scope<'scope, 'env>, run: &'env Run, id: usize) {
    // Each thread starts its children before doing its own work, so that 12 threads are
    // running after a few rounds of spawning, not after twelve.
    for child in (2 * id + 1..2 * id + 3).filter(|&c| c < run.shares.len()) {
        scope.spawn(move || spawn_tree(scope, run, child));
    }
    unsafe { run.work(id) };
}

fn main() {
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    unsafe {
        // Mapped rather than read: a few fault-arounds instead of a copy into fresh pages.
        let [text, queries] = ["/data/docs.txt", "/data/queries.txt"].map(|path| map_file(path));
        // The last line's id gives the document count.
        let last_line = text.rsplit(|&c| c == b'\n').nth(1);
        let docs = last_line.map_or(0, |line| id_of(line.as_ptr()) + 1);
        assert!(docs < 1 << DOC_BITS);
        // Even an empty corpus has one phase, so that the answers are written the usual way.
        let phases = text.len().div_ceil(PHASE_CLAIMS * CLAIM).max(1);
        // Phases start on documents that are multiples of 64, and a bitmap covers one phase.
        let phase_base: Vec<u32> = (0..=phases)
            .map(|phase| aligned_line(text, docs, phase * PHASE_CLAIMS * CLAIM).1 as u32)
            .collect();
        let widest = phase_base.windows(2).fold(0, |m, w| m.max(w[1] - w[0]));
        // A multiple of 4 words, because `add_bits` reads 256 bits at a time.
        let words_per_bitmap = (widest as usize / 64 + 2).next_multiple_of(4);
        // A batch is 64 queries of at least 7 bytes, and each share may end in a short one.
        let batches = queries.len() / (7 * QUERY_BATCH) + threads;
        let run = Run {
            text,
            docs,
            words_per_bitmap,
            buckets: reserve(BUCKETS, true),
            terms: reserve(MAX_TERMS, true),
            term_count: AtomicUsize::new(0),
            queries,
            shares: (0..threads).map(|_| OnceLock::new()).collect(),
            phrases: reserve(MAX_TERMS, false),
            sample: space(MAX_TERMS),
            setup: [0; 6].map(AtomicUsize::new),
            phase: (0..phases).map(|_| Default::default()).collect(),
            phase_base,
            arrived: Default::default(),
            round: Default::default(),
            bitmaps: reserve(2 * MAX_HEAVY * words_per_bitmap, true),
            // Each shard has more rows than it can hold, `len + 4096`: only touched pages cost memory.
            rows: reserve(SHARDS * ((text.len() + 4096 + 15) & !15), false),
            claimed: (0..SHARDS).map(|_| AtomicUsize::new(0)).collect(),
            postings: reserve(text.len() + HUGE_PAGE, true),
            parts: (0..batches).map(|_| Mutex::default()).collect(),
            workspace: reserve(threads * WORKSPACE, false),
        };
        // Huge pages are wanted for the bucket table alone: touch it so it is faulted in as one,
        // then forbid them. Where the kernel gives every region huge pages (THP `always`), thread
        // stacks, heaps and the half-used postings and bitmaps would otherwise each be charged
        // 2 MiB a page; four-kilobyte pages on those cost a few extra faults and nothing more.
        run.buckets.cast::<u8>().write_volatile(0);
        prctl(41, 1, 0, 0, 0);

        thread::scope(|s| spawn_tree(s, &run, 0));

        let mut out = File::create("/out/results.txt").unwrap();
        for part in &run.parts {
            out.write_all(&part.lock().unwrap()).unwrap();
        }
    }
}

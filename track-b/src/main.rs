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
//! parse a share of `queries.txt` and intern its terms into the shared bucket table, merge
//! (one thread), sample 12 claims, plan anchors and bitmaps, then for each phase scan its
//! 256 KiB claims, build its postings shard by shard, and answer it. The answers of phase
//! k - 1 run on whichever threads have run out of scan claims in phase k, so no thread waits
//! for the slowest scanner. Work is shared through atomic job counters, one set per phase,
//! so no counter is ever reset. `main` writes the answer parts in file order.
//!
//! The term table is 2^15 buckets of one cache line, each with 7 tags and 8 actions
//! (`Bucket`), on a huge page. A token's action says what the scan does with it: nothing,
//! set a bit in a bitmap, write a row for a list, or also start a phrase check.
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex, OnceLock};
use std::{arch::x86_64::*, fs::File, io::Write, os::fd::AsRawFd, ptr::null_mut, thread};

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, arg: usize, a: usize, b: usize, c: usize) -> i32;
    fn memchr(text: *const u8, byte: i32, len: usize) -> *const u8;
}
const PROT_READ: i32 = 1;
const PROT_RW: i32 = 3; // PROT_READ | PROT_WRITE
const ANONYMOUS: i32 = 0x4022; // MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
const FILE_AT: i32 = 0x12; // MAP_PRIVATE | MAP_FIXED: a file over pages already reserved
const MADV_HUGEPAGE: i32 = 14;
const MADV_NOHUGEPAGE: i32 = 15;
const MADV_DONTNEED: i32 = 4;
const PR_SET_THP_DISABLE: i32 = 41;

const HUGE_PAGE: usize = 1 << 21;
const CLAIM: usize = 1 << 18;
const PHASE_CLAIMS: usize = 200; // 50 MiB of text between turning rows into postings
const HEADER: usize = 1; // u16s before a list: a sentinel while it fills, so duplicates drop with no branch
const CLEAR_JOBS: usize = 16; // slices of the idle bitmap set that are zeroed during a phase
const TALLY_IDS: usize = 10;
const BATCH: usize = 1 << 14; // bytes of text between the token pass and the term pass
const MAX_TOKENS: usize = 1 << 19; // of a batch and an unfinished document
const BUCKET_BITS: u32 = 15;
const BUCKETS: usize = 1 << BUCKET_BITS;
const BUCKET_SLOTS: usize = 7;
const ID_BLOCK: usize = 32; // term ids a thread takes at once while parsing queries
const MAX_TERMS: usize = BUCKETS * BUCKET_SLOTS + 64 * ID_BLOCK;
const AHEAD: usize = 20; // tokens between a bucket's prefetch and its lookup
const HEAVY: u32 = 1 << 31; // an action: 0 for no term, a bitmap index here, or a term id + 1
const OVERFLOW: u32 = 1 << 30; // in place of 0: the bucket is full, so look further
const ANCHOR: u32 = 1 << 29; // added to a term's action: some phrase starts its check here
const HEAVY_SHARE: usize = 32; // a term in 1/32 of documents is cheaper as a bitmap
const MAX_HEAVY: usize = 4096;
const SHARDS: usize = 64;
const SHARD_BITS: u32 = 6;
const DOC_BITS: u32 = 18;
const CHUNK: usize = 64; // rows; the first of each chunk names the claim that wrote it
const QUERY_BATCH: usize = 64;
const SEARCH_RATIO: usize = 16; // lists this much shorter are searched for, not marked
const LINES: usize = 6; // cache lines of a list requested before a query reads it
const PLAN_JOBS: usize = 16;
const SAMPLE_CLAIMS: usize = 12; // claims read to choose the terms that get bitmaps
const WORKSPACE: usize = 32 << 20; // a worker's share of the one reservation for batch space

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
    heavy_hits: &'static mut [u32],  // bitmap << DOC_BITS | document
    list_hits: &'static mut [u64],   // action << 32 | document
    anchor_hits: &'static mut [u64], // term + 1 << 40 | document << 20 | token (the shift drops the action's flags)
    counts: &'static mut [u32],      // in the sample, how often each term occurs
}

#[derive(Clone, Copy)]
struct Term {
    first: u32,   // the header slot of the term's list in this phase, or its bitmap index
    count: u32,   // documents containing the term
    phrases: u32, // the first phrase anchored here, + 1
    heavy: bool,
    in_bool: bool,
}

/// A phrase, its words in `words` and `word_terms`, and the pseudo-term holding its matches.
struct Phrase {
    term: u32,
    first: u32,
    len: u32,
    pivot: u32, // which word is the anchor, chosen from the sample
    next: u32,  // the next phrase with the same anchor, + 1
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

struct Scratch {
    lists: [Vec<u16>; 2],
    active: usize, // which list the next result goes into
    bits: Vec<u64>,
    marks: Vec<u64>, // all zeros between uses
}

/// What the phases so far found for one query. All zeros is the empty answer, so the array
/// of them can come straight from fresh pages.
#[derive(Clone, Copy)]
struct Tally {
    count: u32,
    sum: u64,
    first: [u32; TALLY_IDS], // the smallest ids, filled from the first phases on
}

impl Tally {
    /// Adds a phase's result as a list of ids relative to the phase's first document.
    fn add_list(&mut self, base: u32, list: &[u16]) {
        let room = TALLY_IDS.saturating_sub(self.count as usize);
        for (slot, &doc) in self.first[TALLY_IDS - room..].iter_mut().zip(list) {
            *slot = base + doc as u32;
        }
        self.count += list.len() as u32;
        self.sum +=
            list.iter().map(|&doc| doc as u64).sum::<u64>() + base as u64 * list.len() as u64;
    }

    /// Adds a phase's result as a bitmap, whose length is a multiple of 4 words. The sum of the
    /// ids in a byte is 8 * its index * its popcount, plus the sum of its bit positions, and
    /// both come from nibble lookups, 32 bytes at a time.
    fn add_bits(&mut self, base: u32, bits: &[u64]) {
        let (mut count, mut sum) = (0, 0);
        unsafe {
            let table =
                |row: [i8; 16]| _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast()));
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
                let high_ones = _mm256_shuffle_epi8(ones, hi);
                let per_byte = _mm256_add_epi8(_mm256_shuffle_epi8(ones, lo), high_ones);
                let low_places = _mm256_add_epi8(
                    _mm256_shuffle_epi8(places, lo),
                    _mm256_shuffle_epi8(places, hi),
                );
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
                std::mem::transmute::<_, [u32; 8]>(v)
                    .map(u64::from)
                    .iter()
                    .sum::<u64>()
            };
            count = total(counts);
            sum = total(sums) + total(starts) + total(offsets);
        }
        // The smallest ids, if the tally has room for more.
        let ids = bits
            .iter()
            .enumerate()
            .filter(|(_, &word)| word != 0)
            .flat_map(|(index, &word)| {
                (0..64)
                    .filter(move |bit| word >> bit & 1 == 1)
                    .map(move |bit| index as u32 * 64 + bit)
            });
        let room = (self.count as usize).min(TALLY_IDS);
        for (slot, id) in self.first[room..].iter_mut().zip(ids) {
            *slot = base + id;
        }
        self.count += count as u32;
        self.sum += sum + base as u64 * count;
    }

    fn write(&self, out: &mut Vec<u8>) {
        put_number(out, self.count as u64);
        out.push(b' ');
        put_number(out, self.sum);
        for &doc in &self.first[..(self.count as usize).min(TALLY_IDS)] {
            out.push(b' ');
            put_number(out, doc as u64);
        }
        out.push(b'\n');
    }
}

struct Run {
    text: *const u8,
    len: usize,
    docs: usize,
    words_per_bitmap: usize,
    buckets: *mut Bucket,
    terms: *mut Term,
    term_count: AtomicUsize,
    query_text: *const u8,
    query_len: usize,
    parsed: Vec<Mutex<Parsed>>,
    merged: OnceLock<Merged>,
    phrases: *mut Phrase,
    sample: &'static [AtomicU32], // per term, the documents of the sample that hold it
    sample_docs: AtomicUsize,
    next_claim: AtomicUsize,
    // One counter of claims and one of jobs per phase, so that no phase needs resetting.
    phase_claims: Vec<AtomicUsize>,
    phase_jobs: Vec<AtomicUsize>,
    // Per phase: the jobs that build, the jobs that answer, and the next free posting.
    answer_jobs: Vec<AtomicUsize>,
    phase_postings: Vec<AtomicUsize>,
    phase_base: Vec<u32>, // the first document of each phase, then the document count
    heavy_total: AtomicUsize,
    plan_jobs: AtomicUsize,
    bucket_jobs: AtomicUsize,
    step: Barrier,
    bitmaps: *mut u64,
    rows: *mut u32,
    stride: usize,
    claimed: Vec<AtomicUsize>,
    postings: *mut u16,
    parts: Mutex<Vec<Vec<u8>>>,
    workspace: *mut u8,
}
unsafe impl Sync for Run {}

unsafe fn reserve(bytes: usize, advice: i32) -> *mut u8 {
    let region = mmap(null_mut(), bytes + HUGE_PAGE, PROT_RW, ANONYMOUS, -1, 0);
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, bytes, advice);
    aligned
}

/// Lazily backed space on small pages, so that large bounds cost only what is touched.
unsafe fn space<T>(count: usize) -> &'static mut [T] {
    let region = reserve(count * size_of::<T>(), MADV_NOHUGEPAGE);
    std::slice::from_raw_parts_mut(region.cast(), count)
}

/// Hands out the parts of one worker's share of a reservation made before the threads start.
/// One `mmap` for all the workers, instead of eight each: every `mmap` and `madvise` takes the
/// address-space lock for writing, which stalls the page faults of the threads already running.
struct Carve(*mut u8, *mut u8);

impl Carve {
    unsafe fn take<T>(&mut self, count: usize) -> &'static mut [T] {
        let at = self.0;
        self.0 = at.add((count * size_of::<T>()).next_multiple_of(4096));
        assert!(self.0 <= self.1);
        std::slice::from_raw_parts_mut(at.cast(), count)
    }
}

unsafe fn map_file(file: &File, len: usize) -> *const u8 {
    // The file goes over the start of a longer reservation, so that a vector load past its end
    // reads zeros. An empty file fails to map and leaves just the zeros.
    let span = len.div_ceil(4096) * 4096 + 4096;
    let base = mmap(null_mut(), span, PROT_READ, ANONYMOUS, -1, 0);
    mmap(base, len, PROT_READ, FILE_AT, file.as_raw_fd(), 0);
    base
}

/// Multiplies to 128 bits and folds, the mixing step of wyhash.
#[inline(always)]
fn fold_multiply(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
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

#[inline(always)]
unsafe fn hash_of(token: *const u8, len: usize) -> u64 {
    if len > 16 {
        return long_hash(token, len);
    }
    let mask = _mm_loadu_si128(MASKS.as_flattened().as_ptr().add(16 - len).cast());
    let folded = _mm_or_si128(_mm_loadu_si128(token.cast()), _mm_set1_epi8(0x20));
    let key = _mm_and_si128(folded, mask);
    // Two AES rounds spread every byte over the whole block, in fewer uops than a multiply.
    let mixed = _mm_aesenc_si128(_mm_aesenc_si128(key, _mm_set1_epi64x(KEY)), key);
    (_mm_cvtsi128_si64(mixed) ^ _mm_extract_epi64(mixed, 1)) as u64 | 1
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
    (hash >> (64 - BUCKET_BITS)) as usize
}

#[cold]
unsafe fn long_hash(token: *const u8, len: usize) -> u64 {
    let bytes = std::slice::from_raw_parts(token, len);
    bytes.iter().fold(len as u64, |hash, &c| {
        fold_multiply(hash ^ (c | 0x20) as u64, 0xE703_7ED1_A0B4_28DB)
    }) | 1
}

impl Run {
    /// The position and id of the first line at or after `at` whose id is a multiple of 64.
    /// Claims start on such lines, so no two workers ever share a word of a bitmap.
    unsafe fn aligned_line(&self, at: usize) -> Option<(usize, usize)> {
        let past_line = |from: usize| {
            let found = memchr(self.text.add(from), b'\n' as i32, self.len - from);
            found as usize - self.text as usize + 1
        };
        let id_of = |line: usize| {
            (line..)
                .map(|i| *self.text.add(i))
                .take_while(|&c| c != b'\t')
                .fold(0, |id, digit| id * 10 + (digit - b'0') as usize)
        };
        let mut line = at;
        if at > 0 && at < self.len && *self.text.add(at - 1) != b'\n' {
            line = past_line(at);
        }
        while line < self.len {
            let id = id_of(line);
            if id % 64 == 0 {
                return Some((line, id));
            }
            line = past_line(line);
        }
        None
    }

    /// The action of a token whose bucket is full, so that its term may be in a later one.
    #[cold]
    unsafe fn action_past(buckets: *const Bucket, hash: u64) -> u32 {
        let mut index = bucket_of(hash);
        loop {
            let bucket = &*buckets.add(index);
            for slot in 0..BUCKET_SLOTS {
                if bucket.tags[slot] == hash as u32 {
                    return bucket.actions[slot];
                }
            }
            if bucket.actions[BUCKET_SLOTS] == 0 {
                return 0;
            }
            index = (index + 1) % BUCKETS;
        }
    }

    #[cold]
    unsafe fn claim_chunk(&self, shard: usize, claim: usize) -> (*mut u32, *mut u32) {
        let offset = self.claimed[shard].fetch_add(CHUNK, Relaxed);
        let at = self.rows.add(shard * self.stride + offset);
        *at = claim as u32 + 1;
        (at.add(1), at.add(CHUNK))
    }

    /// Reads one claim. With SAMPLE set, it only counts how often each term appears.
    unsafe fn index_claim<const SAMPLE: bool>(&self, claim: usize, b: &mut Buffers) -> usize {
        let Some((start, mut doc)) = self.aligned_line(claim * CLAIM) else {
            return 0;
        };
        let end = self
            .aligned_line((claim + 1) * CLAIM)
            .map_or(self.len, |line| line.0);
        if start >= end {
            return 0;
        }
        let first_doc = doc;
        let text = self.text;

        let mut cursors = [(null_mut(), null_mut()); SHARDS];
        let (mut token_count, mut end_count, mut line_count) = (0, 0, 0);
        let mut carry = 0u64; // whether the previous 64 bytes ended inside a token
        let mut at = start;
        let mut first_token = true;

        let floor_lower = _mm256_set1_epi8(b'a' as i8 - 1);
        let ceil_lower = _mm256_set1_epi8(b'z' as i8 + 1);
        let floor_digit = _mm256_set1_epi8(b'0' as i8 - 1);
        let ceil_digit = _mm256_set1_epi8(b'9' as i8 + 1);
        let fold = _mm256_set1_epi8(0x20);
        let newline = _mm256_set1_epi8(b'\n' as i8);

        while at < end {
            let batch_end = (at + BATCH).min(end);
            while at < batch_end {
                let classify = |at: usize| {
                    let chunk = _mm256_loadu_si256(text.add(at).cast());
                    let lower = _mm256_or_si256(chunk, fold);
                    let letters = _mm256_and_si256(
                        _mm256_cmpgt_epi8(lower, floor_lower),
                        _mm256_cmpgt_epi8(ceil_lower, lower),
                    );
                    let digits = _mm256_and_si256(
                        _mm256_cmpgt_epi8(chunk, floor_digit),
                        _mm256_cmpgt_epi8(ceil_digit, chunk),
                    );
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
                let (found, finished) =
                    (opened.count_ones() as usize, closed.count_ones() as usize);
                for extra in 0..12 {
                    *b.starts.get_unchecked_mut(token_count + extra) =
                        (at + opened.trailing_zeros() as usize) as u32;
                    *b.ends.get_unchecked_mut(end_count + extra) =
                        (at + closed.trailing_zeros() as usize) as u32;
                    opened &= opened.wrapping_sub(1);
                    closed &= closed.wrapping_sub(1);
                }
                let mut extra = 12;
                while opened != 0 {
                    b.starts[token_count + extra] = (at + opened.trailing_zeros() as usize) as u32;
                    opened &= opened - 1;
                    extra += 1;
                }
                extra = 12;
                while closed != 0 {
                    b.ends[end_count + extra] = (at + closed.trailing_zeros() as usize) as u32;
                    closed &= closed - 1;
                    extra += 1;
                }
                token_count += found;
                end_count += finished;
                at += 64;
            }

            // Only whole documents go on, so that a phrase has all its tokens at hand.
            let ready = if line_count > 0 {
                b.lines[line_count - 1] as usize
            } else {
                0
            };
            let mut token = 0;
            if first_token && ready > 0 {
                first_token = false;
                token = 1; // the id of the claim's first document
                b.hashes[0] = 0;
            }
            let mut i = token;
            while i + 1 < ready {
                let (from_a, to_a) = (
                    *b.starts.get_unchecked(i) as usize,
                    *b.ends.get_unchecked(i) as usize,
                );
                let (from_b, to_b) = (
                    *b.starts.get_unchecked(i + 1) as usize,
                    *b.ends.get_unchecked(i + 1) as usize,
                );
                let (len_a, len_b) = (to_a - from_a, to_b - from_b);
                if (len_a | len_b) > 16 {
                    *b.hashes.get_unchecked_mut(i) = hash_of(text.add(from_a), len_a);
                    *b.hashes.get_unchecked_mut(i + 1) = hash_of(text.add(from_b), len_b);
                } else {
                    let masks = MASKS.as_flattened().as_ptr();
                    let pair = _mm256_set_m128i(
                        _mm_loadu_si128(text.add(from_b).cast()),
                        _mm_loadu_si128(text.add(from_a).cast()),
                    );
                    let mask = _mm256_set_m128i(
                        _mm_loadu_si128(masks.add(16 - len_b).cast()),
                        _mm_loadu_si128(masks.add(16 - len_a).cast()),
                    );
                    let key = _mm256_and_si256(_mm256_or_si256(pair, _mm256_set1_epi8(0x20)), mask);
                    let mixed = _mm256_aesenc_epi128(
                        _mm256_aesenc_epi128(key, _mm256_set1_epi64x(KEY)),
                        key,
                    );
                    let folded = _mm256_xor_si256(mixed, _mm256_srli_si256::<8>(mixed));
                    *b.hashes.get_unchecked_mut(i) = _mm256_extract_epi64::<0>(folded) as u64 | 1;
                    *b.hashes.get_unchecked_mut(i + 1) =
                        _mm256_extract_epi64::<2>(folded) as u64 | 1;
                }
                i += 2;
            }
            if i < ready {
                let (from, to) = (
                    *b.starts.get_unchecked(i) as usize,
                    *b.ends.get_unchecked(i) as usize,
                );
                *b.hashes.get_unchecked_mut(i) = hash_of(text.add(from), to - from);
            }
            let (mut heavy_count, mut list_count, mut anchor_count) = (0, 0, 0);
            // Local copies of the pointers: through `self` and `b` the compiler reloads them
            // after every store, as a store might have changed them.
            let (buckets, hashes) = (self.buckets, b.hashes.as_ptr());
            let (heavy_hits, list_hits) = (b.heavy_hits.as_mut_ptr(), b.list_hits.as_mut_ptr());
            for line in 0..line_count {
                let to = b.lines[line] as usize;
                while token < to {
                    let ahead = bucket_of(*hashes.add(token + AHEAD));
                    _mm_prefetch::<_MM_HINT_T0>(buckets.add(ahead).cast());
                    let action = action_in(buckets, *hashes.add(token));
                    // All three stores happen, and only the right counts move.
                    *heavy_hits.add(heavy_count) = action << DOC_BITS | doc as u32;
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
                    let (first, len) = (phrase.first as usize, phrase.len as usize);
                    let pivot = phrase.pivot as usize;
                    if token >= pivot
                        && token - pivot + len <= ready
                        && b.hashes[token - pivot..token - pivot + len]
                            == self.merged().words[first..first + len]
                    {
                        b.list_hits[list_count] =
                            (phrase.term as u64 + 1) << 32 | (hit >> 20 & ((1 << 20) - 1));
                        list_count += 1;
                    }
                }
            }
            if heavy_count > 0 {
                let phase = claim / PHASE_CLAIMS;
                let (set, base) = (self.bitmap(0, phase), self.phase_base[phase]);
                let width = self.words_per_bitmap;
                for &hit in &b.heavy_hits[..heavy_count] {
                    let doc = (hit & ((1 << DOC_BITS) - 1)) - base;
                    let at = (hit >> DOC_BITS) as usize * width + (doc / 64) as usize;
                    *set.add(at) |= 1 << (doc % 64);
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
                if cursor.0 == cursor.1 {
                    *cursor = self.claim_chunk(shard, claim);
                }
                *cursor.0 = ((term >> SHARD_BITS) + 1) << DOC_BITS | hit as u32;
                cursor.0 = cursor.0.add(1);
            }
            // A newline left over belongs to a document whose id comes in the next batch.
            if line_count > 0 {
                b.lines[0] = 0;
                line_count = 1;
            }
            b.starts.copy_within(ready..token_count, 0);
            token_count -= ready;
            b.ends.copy_within(ready..end_count, 0);
            end_count -= ready;
        }
        // Drop the scanned pages from this process's page tables now, on every thread, instead
        // of one thread tearing down the whole mapping at exit. The mapping is a read-only
        // private file, so a later touch (a neighbour's line search) just maps the page again.
        // The sample's claims are read again by the main pass, so they are kept.
        if !SAMPLE {
            let (from, to) = (start.div_ceil(4096) * 4096, end / 4096 * 4096);
            if from < to {
                madvise(text.add(from) as *mut u8, to - from, MADV_DONTNEED);
            }
        }
        // The documents this claim held, which the sample uses to scale its counts.
        doc - first_doc + 1
    }

    fn each_job(&self, counter: &AtomicUsize, count: usize, mut job: impl FnMut(usize)) {
        loop {
            let next = counter.fetch_add(1, Relaxed);
            if next >= count {
                return;
            }
            job(next);
        }
    }

    /// Uses the sample to give each phrase in a slice its rarest word as an anchor. Phrases
    /// anchored on one term are chained through the term by an atomic swap, in any order.
    unsafe fn plan_phrases(&self, slice: usize) {
        let merged = self.merged();
        for index in
            merged.phrase_count * slice / PLAN_JOBS..merged.phrase_count * (slice + 1) / PLAN_JOBS
        {
            let phrase = &mut *self.phrases.add(index);
            let first = phrase.first as usize;
            let terms = &merged.word_terms[first..first + phrase.len as usize];
            let pivot = (0..terms.len())
                .min_by_key(|&i| self.sample[terms[i] as usize].load(Relaxed))
                .unwrap();
            let anchor = &mut (*self.terms.add(terms[pivot] as usize)).phrases;
            phrase.pivot = pivot as u32;
            phrase.next = AtomicU32::from_ptr(anchor).swap(index as u32 + 1, Relaxed);
        }
    }

    /// Gives the actions in a slice of the buckets their final form, once the phrases have been
    /// planned. Slices are taken by whichever thread is free.
    unsafe fn plan_buckets(&self, slice: usize) {
        let docs = self.sample_docs.load(Relaxed);
        for index in BUCKETS * slice / PLAN_JOBS..BUCKETS * (slice + 1) / PLAN_JOBS {
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
                } else if self.sample[id].load(Relaxed) as usize * HEAVY_SHARE >= docs {
                    let heavy = self.heavy_total.fetch_add(1, Relaxed);
                    if heavy < MAX_HEAVY {
                        *action = HEAVY | heavy as u32;
                        *term = Term {
                            heavy: true,
                            first: heavy as u32,
                            ..*term
                        };
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
        let region = self.rows.add(shard * self.stride);
        let claimed = self.claimed[shard].load(Relaxed);
        let mut chunks: Vec<(u32, u32)> = (0..claimed)
            .step_by(CHUNK)
            .map(|at| (*region.add(at), at as u32))
            .collect();
        chunks.sort_unstable();
        let term_count = self.term_count.load(Relaxed);
        let terms = term_count.div_ceil(SHARDS) + 1;
        // Lists belong to one phase: last phase's are gone, bitmap terms keep their index.
        for id in (shard..term_count).step_by(SHARDS) {
            let info = &mut *self.terms.add(id);
            if !info.heavy {
                (info.first, info.count) = (0, 0);
            }
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
        let mut at = self.phase_postings[phase].fetch_add(total, Relaxed) as u32;
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
    }

    /// Zeroes a slice of the bitmap set that the next phase will fill.
    unsafe fn clear_bitmaps(&self, phase: usize, slice: usize) {
        let heavy = self.heavy_total.load(Relaxed).min(MAX_HEAVY);
        let (from, to) = (heavy * slice / CLEAR_JOBS, heavy * (slice + 1) / CLEAR_JOBS);
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
        let info = *self.terms.add(term as usize);
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
    unsafe fn answer_batch(&self, batch: usize, phase: usize, scratch: &mut Scratch) {
        let base = self.phase_base[phase];
        let (share, first) = self.merged().units[batch];
        let share = &self.merged().shares[share as usize];
        let first = first as usize;
        let end = (first + QUERY_BATCH).min(share.queries.len());
        let tallies = share.tallies.as_ptr() as *mut Tally;
        let steps_of = |query: &std::ops::Range<usize>| &share.steps[query.clone()];
        // Term records and postings are scattered, so a query's records are requested two
        // queries before they are read and the start of its lists one query before.
        let prefetch = |at: usize, lists: bool| {
            let query = share.queries.get(at).filter(|_| at < end);
            for &(_, term) in query.map_or(&[][..], steps_of) {
                let info = &*self.terms.add(term as usize);
                if !lists {
                    _mm_prefetch::<_MM_HINT_T0>((info as *const Term).cast());
                } else if !info.heavy {
                    let list = self.postings.add(info.first as usize);
                    for line in 0..(info.count as usize).div_ceil(32).min(LINES) {
                        _mm_prefetch::<_MM_HINT_T0>(list.add(line * 32).cast());
                    }
                }
            }
        };
        prefetch(first, false);
        prefetch(first + 1, false);
        for (query, index) in share.queries[first..end].iter().zip(first..) {
            prefetch(index + 2, false);
            prefetch(index + 1, true);
            let steps = steps_of(query);
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
            self.parts.lock().unwrap()[batch] = out;
        }
    }

    fn merged(&self) -> &Merged {
        self.merged.get().unwrap()
    }

    /// Hashes and interns the queries of this thread's share of the file, which starts and ends
    /// at lines: share `k` begins at the first line start at or after `k / threads`.
    unsafe fn parse_share(&self, k: usize) -> Parsed {
        let bytes = std::slice::from_raw_parts(self.query_text, self.query_len + 64);
        let (size, shares) = (self.query_len, self.parsed.len());
        let start = |k: usize| match k * size / shares {
            _ if k == shares => size,
            0 => 0,
            at => bytes[at - 1..size]
                .iter()
                .position(|&c| c == b'\n')
                .map_or(size, |n| at + n),
        };
        parse_queries(self, start(k), start(k + 1))
    }

    /// Lists the shares' work units and joins the words of phrases, which are few. Runs on one
    /// thread while the others wait.
    unsafe fn merge(&self) {
        let mut all = Merged::default();
        for (index, share) in self.parsed.iter().enumerate() {
            let mut share = std::mem::take(&mut *share.lock().unwrap());
            let words = all.words.len() as u32;
            for phrase in std::mem::take(&mut share.phrases) {
                *self.phrases.add(all.phrase_count) = Phrase {
                    first: phrase.first + words,
                    ..phrase
                };
                all.phrase_count += 1;
            }
            all.words.append(&mut share.words);
            all.word_terms.append(&mut share.word_terms);
            all.units.extend(
                (0..share.queries.len())
                    .step_by(QUERY_BATCH)
                    .map(|first| (index as u32, first as u32)),
            );
            all.shares.push(share);
        }
        *self.parts.lock().unwrap() = vec![Vec::new(); all.units.len()];
        self.merged.set(all).ok();
    }

    unsafe fn work(&self, id: usize) {
        let tokens = MAX_TOKENS + 32;
        let share = self.workspace.add(id * WORKSPACE);
        let mut carve = Carve(share, share.add(WORKSPACE));
        let mut buffers = Buffers {
            starts: carve.take(tokens),
            ends: carve.take(tokens),
            lines: carve.take(BATCH / 2 + 64),
            hashes: carve.take(tokens + AHEAD),
            heavy_hits: carve.take(tokens),
            list_hits: carve.take(tokens),
            anchor_hits: carve.take(tokens),
            counts: carve.take(MAX_TERMS),
        };
        *self.parsed[id].lock().unwrap() = self.parse_share(id);
        if self.step.wait().is_leader() {
            self.merge();
        }
        self.step.wait();
        // Threads that wake late find fewer claims left, instead of delaying everyone.
        let mut docs = 0;
        self.each_job(&self.next_claim, SAMPLE_CLAIMS, |claim| {
            docs += self.index_claim::<true>(claim, &mut buffers)
        });
        let terms = self.term_count.load(Relaxed);
        for (total, &part) in self.sample.iter().zip(&buffers.counts[..terms]) {
            total.fetch_add(part, Relaxed);
        }
        self.sample_docs.fetch_add(docs, Relaxed);
        // The counts are done with: give their pages back, so they do not add to the peak.
        madvise(
            buffers.counts.as_mut_ptr().cast(),
            MAX_TERMS * 4,
            MADV_DONTNEED,
        );
        self.step.wait();
        self.each_job(&self.plan_jobs, PLAN_JOBS, |slice| self.plan_phrases(slice));
        self.step.wait();
        self.each_job(&self.bucket_jobs, PLAN_JOBS, |part| self.plan_buckets(part));
        self.step.wait();

        let batches = self.merged().units.len();
        let mut scratch = Scratch {
            lists: [Vec::new(), Vec::new()],
            active: 0,
            bits: vec![0; self.words_per_bitmap],
            marks: vec![0; self.words_per_bitmap],
        };
        let phases = self.phase_base.len() - 1;
        // Stage k scans phase k, and a thread that runs out of claims answers phase k - 1
        // while the others finish scanning: the postings and bitmaps of k - 1 are still
        // there, because the build of k comes after the barrier that ends every answer.
        for stage in 0..=phases {
            if stage < phases {
                // A claim past the end of the text finds no line to start on and does nothing.
                self.each_job(&self.phase_claims[stage], PHASE_CLAIMS, |claim| {
                    self.index_claim::<false>(stage * PHASE_CLAIMS + claim, &mut buffers);
                });
            }
            if stage > 0 {
                self.each_job(&self.answer_jobs[stage - 1], batches, |batch| {
                    self.answer_batch(batch, stage - 1, &mut scratch)
                });
            }
            if stage == phases {
                break;
            }
            self.step.wait();
            // Postings are u16 offsets from the phase's first document, with u16::MAX the sentinel.
            assert!(self.phase_base[stage + 1] - self.phase_base[stage] < u16::MAX as u32);
            self.each_job(&self.phase_jobs[stage], SHARDS + CLEAR_JOBS, |job| {
                if job < SHARDS {
                    self.build_shard(job, stage)
                } else {
                    self.clear_bitmaps(stage, job - SHARDS)
                }
            });
            self.step.wait();
        }
    }
}

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

/// How many of `list`, which is sorted, are below `value`. The search runs the same number of
/// steps whatever the value, and each step is a conditional move, so there is no branch to
/// mispredict and the searches for neighbouring values overlap.
#[inline(always)]
unsafe fn below(list: &[u16], value: u16) -> usize {
    let (mut base, mut size) = (0, list.len());
    while size > 1 {
        let half = size / 2;
        base += (*list.get_unchecked(base + half) < value) as usize * half;
        size -= half;
    }
    base + (size > 0 && *list.get_unchecked(base) < value) as usize
}

/// Appends the values of `from` that are in `other` (or, with `keep` false, are not). A list
/// much shorter than the other is searched for in it. Otherwise one list is marked in `marks`,
/// which is all zeros between calls, and the other is tested against the marks and the marks
/// cleared again. Both write every value and move the count only for a kept one, so neither
/// branches on the data.
unsafe fn filter_by(
    from: &[u16],
    other: &[u16],
    keep: bool,
    out: &mut Vec<u16>,
    marks: &mut [u64],
) {
    out.reserve(from.len().max(other.len()));
    let target = out.as_mut_ptr();
    let mut found = 0;
    if from.len() * SEARCH_RATIO <= other.len() {
        for &doc in from {
            let at = below(other, doc);
            let present = at < other.len() && *other.get_unchecked(at) == doc;
            *target.add(found) = doc;
            found += (present == keep) as usize;
        }
    } else {
        let (marked, tested) = if keep { (from, other) } else { (other, from) };
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
    let (a, b) = if swap {
        (term, current)
    } else {
        (current, term)
    };
    scratch.active ^= 1;
    let out = &mut scratch.lists[scratch.active];
    out.clear();
    let bits = &mut scratch.bits;
    let words = bits.len();
    let list = |out: &Vec<u16>| Set::List(std::slice::from_raw_parts(out.as_ptr(), out.len()));
    match (op, a, b) {
        (Op::And | Op::Not, Set::List(a), Set::List(b)) => {
            filter_by(a, b, op == Op::And, out, &mut scratch.marks);
            list(out)
        }
        (Op::And | Op::Not, Set::List(a), Set::Bits(b)) => {
            // Keep what the bitmap holds for AND, and what it lacks for NOT.
            let held = |doc: u16| *b.get_unchecked(doc as usize / 64) >> (doc % 64) & 1 == 1;
            out.extend(a.iter().filter(|&&doc| held(doc) == (op == Op::And)));
            list(out)
        }
        // A merged list is cheaper than a bitmap while it is shorter than one.
        (Op::Or, Set::List(a), Set::List(b)) if a.len() + b.len() < words => {
            union(a, b, out);
            list(out)
        }
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
            Set::Bits(std::slice::from_raw_parts(bits.as_ptr(), words))
        }
    }
}

fn put_number(out: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0u8; 20];
    let mut at = digits.len();
    loop {
        at -= 1;
        digits[at] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    out.extend_from_slice(&digits[at..]);
}

/// Adds a term to the table, or finds it, and returns its id. Every thread interns at once:
/// a slot is claimed by a compare-exchange on its tag, and whoever loses to the same tag waits
/// for the winner's id, which is stored right after.
unsafe fn intern(run: &Run, ids: &mut (u32, u32), hash: u64) -> u32 {
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
                loop {
                    match action.load(Acquire) {
                        0 => std::hint::spin_loop(),
                        found => return found - 1,
                    }
                }
            }
        }
        AtomicU32::from_ptr(&mut bucket.actions[BUCKET_SLOTS]).store(OVERFLOW, Relaxed);
        index = (index + 1) % BUCKETS;
    }
}

/// A fresh term id. Ids come in blocks, one atomic add per block; an unused id is a term
/// nothing mentions.
unsafe fn new_id(run: &Run, ids: &mut (u32, u32)) -> u32 {
    if ids.0 == ids.1 {
        let first = run.term_count.fetch_add(ID_BLOCK, Relaxed) as u32;
        *ids = (first, first + ID_BLOCK as u32);
    }
    ids.0 += 1;
    ids.0 - 1
}

/// One thread's share of the queries, with offsets that start at zero in each share.
#[derive(Default)]
struct Parsed {
    queries: Vec<std::ops::Range<usize>>, // each one's steps
    steps: Vec<(Op, u32)>,
    tallies: &'static mut [Tally],
    words: Vec<u64>,
    word_terms: Vec<u32>,
    phrases: Vec<Phrase>,
}

/// Parses and interns the queries in `from..to`, which start and end at lines.
unsafe fn parse_queries(run: &Run, from: usize, to: usize) -> Parsed {
    let text = run.query_text;
    let (mut parsed, mut ids, mut at) = (Parsed::default(), (0, 0), from);
    while at < to {
        let phrase = *text.add(at) == b'P';
        at += if phrase { 8 } else { 5 }; // past `PHRASE "` or `BOOL `
        let (first, first_word) = (parsed.steps.len(), parsed.words.len());
        let mut op = Op::And;
        loop {
            let word = at;
            // Spaces, quotes and newlines all sort below the term bytes.
            while *text.add(at) >= b'0' {
                at += 1;
            }
            let hash = hash_of(text.add(word), at - word);
            let term = intern(run, &mut ids, hash);
            if phrase {
                parsed.words.push(hash);
                parsed.word_terms.push(term);
            } else {
                (*run.terms.add(term as usize)).in_bool = true;
                parsed.steps.push((op, term));
            }
            let stop = *text.add(at);
            at += 1;
            if stop != b' ' {
                at += phrase as usize; // the newline after the closing quote
                break;
            }
            if !phrase {
                (op, at) = match *text.add(at) {
                    b'A' => (Op::And, at + 4),
                    b'O' => (Op::Or, at + 3),
                    _ => (Op::Not, at + 4),
                };
            }
        }
        if phrase {
            let term = new_id(run, &mut ids);
            parsed.steps.push((Op::And, term));
            parsed.phrases.push(Phrase {
                term,
                first: first_word as u32,
                len: (parsed.words.len() - first_word) as u32,
                pivot: 0,
                next: 0,
            });
        }
        parsed.queries.push(first..parsed.steps.len());
    }
    parsed.tallies = space(parsed.queries.len());
    parsed
}

/// Everything the threads parsed. A share keeps its own queries, steps and tallies; the
/// answers go through `units`, runs of up to QUERY_BATCH queries of one share, in file order.
#[derive(Default)]
struct Merged {
    shares: Vec<Parsed>,
    units: Vec<(u32, u32)>,
    words: Vec<u64>,
    word_terms: Vec<u32>,
    phrase_count: usize,
}

fn spawn_tree<'scope, 'env>(scope: &'scope thread::Scope<'scope, 'env>, run: &'env Run, id: usize) {
    // Each thread starts its children before doing its own work, so that 12 threads are
    // running after a few rounds of spawning, not after twelve.
    for child in (2 * id + 1..2 * id + 3).filter(|&c| c < run.parsed.len()) {
        scope.spawn(move || spawn_tree(scope, run, child));
    }
    unsafe { run.work(id) };
}

fn main() {
    let file = File::open("/data/docs.txt").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let queries = File::open("/data/queries.txt").unwrap();
    let query_len = queries.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    // Even an empty corpus has one phase, so that the answers are written the usual way.
    let phases = len.div_ceil(PHASE_CLAIMS * CLAIM).max(1);
    // One counter per phase, each starting at `init`.
    let counters = |init| (0..phases).map(|_| AtomicUsize::new(init)).collect();
    let mut run = unsafe {
        let text = map_file(&file, len);
        // The last line's id gives the document count.
        let body = std::slice::from_raw_parts(text, len);
        let docs = body.rsplit(|&c| c == b'\n').nth(1).map_or(0, |last| {
            let digits = last.iter().take_while(|&&c| c != b'\t');
            digits.fold(0, |id, &digit| id * 10 + (digit - b'0') as usize) + 1
        });
        assert!(docs < 1 << DOC_BITS);
        // Rows per shard: more than any shard can hold. Only touched pages cost memory.
        let stride = len + 4096;
        // Mapped rather than read: a few fault-arounds instead of a copy into fresh pages.
        // The 64 bytes past the end that the parser may look at are zeros, from the page's
        // tail or from the spare page `map_file` leaves.
        let query_text = map_file(&queries, query_len);
        Run {
            text,
            len,
            docs,
            words_per_bitmap: 0,
            buckets: reserve(BUCKETS * size_of::<Bucket>(), MADV_HUGEPAGE).cast(),
            terms: reserve(MAX_TERMS * size_of::<Term>(), MADV_HUGEPAGE).cast(),
            term_count: AtomicUsize::new(0),
            query_text,
            query_len,
            parsed: (0..threads).map(|_| Mutex::default()).collect(),
            merged: OnceLock::new(),
            phrases: reserve(MAX_TERMS * size_of::<Phrase>(), MADV_NOHUGEPAGE).cast(),
            sample: space(MAX_TERMS),
            sample_docs: AtomicUsize::new(0),
            next_claim: AtomicUsize::new(0),
            phase_claims: counters(0),
            phase_jobs: counters(0),
            answer_jobs: counters(0),
            phase_postings: counters(1),
            phase_base: Vec::new(),
            heavy_total: AtomicUsize::new(0),
            plan_jobs: AtomicUsize::new(0),
            bucket_jobs: AtomicUsize::new(0),
            step: Barrier::new(threads),
            bitmaps: null_mut(),
            rows: reserve(SHARDS * stride * 4, MADV_NOHUGEPAGE).cast(),
            stride,
            claimed: (0..SHARDS).map(|_| AtomicUsize::new(0)).collect(),
            postings: reserve(len * 2 + HUGE_PAGE, MADV_HUGEPAGE).cast(),
            parts: Mutex::new(Vec::new()),
            workspace: reserve(threads * WORKSPACE, MADV_NOHUGEPAGE),
        }
    };
    // Phases start on documents that are multiples of 64, and a bitmap covers one phase.
    run.phase_base = (0..=phases)
        .map(|phase| {
            let line = unsafe { run.aligned_line(phase * PHASE_CLAIMS * CLAIM) };
            line.map_or(run.docs, |line| line.1) as u32
        })
        .collect();
    let widest = run
        .phase_base
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0);
    // A multiple of 4 words, because `add_bits` reads 256 bits at a time.
    run.words_per_bitmap = (widest as usize / 64 + 2).next_multiple_of(4);
    run.bitmaps =
        unsafe { reserve(2 * MAX_HEAVY * run.words_per_bitmap * 8, MADV_HUGEPAGE).cast() };
    // Huge pages are wanted for the bucket table alone: touch it so it is faulted in as one,
    // then forbid them. Where the kernel gives every region huge pages (THP `always`), thread
    // stacks, heaps and the half-used postings and bitmaps would otherwise each be charged 2 MiB
    // a page; four-kilobyte pages on those cost a few extra faults and nothing more.
    unsafe {
        run.buckets.cast::<u8>().write_volatile(0);
        prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }

    thread::scope(|s| spawn_tree(s, &run, 0));

    let mut out = File::create("/out/results.txt").unwrap();
    for part in run.parts.lock().unwrap().iter() {
        out.write_all(part).unwrap();
    }
}

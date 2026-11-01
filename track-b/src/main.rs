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
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex, OnceLock};
use std::thread;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, arg: usize, a: usize, b: usize, c: usize) -> i32;
    fn memchr(text: *const u8, byte: i32, len: usize) -> *const u8;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x4000;
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
    anchor_hits: &'static mut [u64], // term + 1 << 40 | document << 20 | token
}

#[derive(Clone, Copy)]
struct Term {
    first: u32,   // the term's last list part, or its bitmap index
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

struct Query {
    first: u32,
    len: u32,
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

    /// Adds a phase's result as a bitmap. The sum of the ids in a byte is 8 * byte index *
    /// popcount plus the sum of the bit positions in it, and both come from nibble lookups, 32
    /// bytes at a time.
    fn add_bits(&mut self, base: u32, bits: &[u64]) {
        let (mut count, mut sum) = (0u64, 0u64);
        unsafe {
            let table =
                |row: [i8; 16]| _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast()));
            let ones = table([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]);
            let places = table([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6]);
            let weights = _mm256_setr_epi8(
                0, 8, 16, 24, 32, 40, 48, 56, 0, 8, 16, 24, 32, 40, 48, 56, 0, 8, 16, 24, 32, 40,
                48, 56, 0, 8, 16, 24, 32, 40, 48, 56,
            );
            let (low, zero) = (_mm256_set1_epi8(15), _mm256_setzero_si256());
            let (mut counts, mut places_sum, mut offsets, mut starts) = (zero, zero, zero, zero);
            let mut first = _mm256_setr_epi64x(0, 64, 128, 192);
            for block in bits.chunks_exact(4) {
                let word = _mm256_loadu_si256(block.as_ptr().cast());
                let (lo, hi) = (
                    _mm256_and_si256(word, low),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(word), low),
                );
                let high_ones = _mm256_shuffle_epi8(ones, hi);
                let per_byte = _mm256_add_epi8(_mm256_shuffle_epi8(ones, lo), high_ones);
                let positions = _mm256_add_epi8(
                    _mm256_add_epi8(
                        _mm256_shuffle_epi8(places, lo),
                        _mm256_shuffle_epi8(places, hi),
                    ),
                    _mm256_slli_epi16::<2>(high_ones),
                );
                let per_word = _mm256_sad_epu8(per_byte, zero);
                counts = _mm256_add_epi64(counts, per_word);
                places_sum = _mm256_add_epi64(places_sum, _mm256_sad_epu8(positions, zero));
                starts = _mm256_add_epi64(starts, _mm256_mul_epu32(per_word, first));
                offsets = _mm256_add_epi32(
                    offsets,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(per_byte, weights),
                        _mm256_set1_epi16(1),
                    ),
                );
                first = _mm256_add_epi64(first, _mm256_set1_epi64x(256));
            }
            let lanes = |v: __m256i| -> [u64; 4] { std::mem::transmute(v) };
            let narrow: [u32; 8] = std::mem::transmute(offsets);
            count += lanes(counts).iter().sum::<u64>();
            sum += lanes(places_sum).iter().chain(&lanes(starts)).sum::<u64>();
            sum += narrow.iter().map(|&n| n as u64).sum::<u64>();
        }
        for (index, &word) in bits.chunks_exact(4).remainder().iter().enumerate() {
            let at = (bits.len() / 4 * 4 + index) as u64 * 64;
            let ones = word.count_ones() as u64;
            count += ones;
            let mut rest = word;
            while rest != 0 {
                sum += at + rest.trailing_zeros() as u64;
                rest &= rest - 1;
            }
        }
        let mut written = self.count as usize;
        'words: for (index, &word) in bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                if written >= TALLY_IDS {
                    break 'words;
                }
                self.first[written] = base + index as u32 * 64 + word.trailing_zeros();
                word &= word - 1;
                written += 1;
            }
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
    samples: Mutex<(Vec<u32>, usize)>,
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
}
unsafe impl Sync for Run {}

unsafe fn reserve(bytes: usize, advice: i32) -> *mut u8 {
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    let span = bytes + HUGE_PAGE;
    let region = mmap(
        std::ptr::null_mut(),
        span,
        PROT_READ | PROT_WRITE,
        flags,
        -1,
        0,
    );
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, bytes, advice);
    aligned
}

/// Lazily backed space on small pages, so that large bounds cost only what is touched.
unsafe fn space<T>(count: usize) -> &'static mut [T] {
    let region = reserve(count * size_of::<T>(), MADV_NOHUGEPAGE);
    std::slice::from_raw_parts_mut(region.cast(), count)
}

unsafe fn map_file(file: &File, len: usize) -> *const u8 {
    let span = len.div_ceil(4096) * 4096 + 4096;
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    let base = mmap(std::ptr::null_mut(), span, PROT_READ, flags, -1, 0);
    if len > 0 {
        mmap(
            base,
            len,
            PROT_READ,
            MAP_PRIVATE | MAP_FIXED,
            file.as_raw_fd(),
            0,
        );
    }
    base
}

/// Multiplies to 128 bits and folds, the mixing step of wyhash.
#[inline(always)]
fn fold_multiply(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

/// The hash of a token in any case, never 0, which marks an unused slot. Tokens are compared
/// by this hash alone: of 600,000 tokens and 80,000 terms, two sharing 64 bits has odds of
/// about 3 in a billion. Every token byte is a letter or digit, and `| 0x20` lowers letters
/// and leaves digits, so folding needs no check on each byte.
const KEY: i64 = 0x9E37_79B9_7F4A_7C15_u64 as i64;

#[repr(align(16))]
struct Masks([u8; 32]);
static MASKS: Masks = {
    let mut bytes = [0u8; 32];
    let mut i = 0;
    while i < 16 {
        bytes[i] = 0xFF;
        i += 1;
    }
    Masks(bytes)
};

#[inline(always)]
unsafe fn hash_of(token: *const u8, len: usize) -> u64 {
    if len > 16 {
        return long_hash(token, len);
    }
    let mask = _mm_loadu_si128(MASKS.0.as_ptr().add(16 - len).cast());
    let folded = _mm_or_si128(_mm_loadu_si128(token.cast()), _mm_set1_epi8(0x20));
    let key = _mm_and_si128(folded, mask);
    // Two AES rounds spread every byte over the whole block, in fewer uops than a multiply.
    let mixed = _mm_aesenc_si128(_mm_aesenc_si128(key, _mm_set1_epi64x(KEY)), key);
    (_mm_cvtsi128_si64(mixed) ^ _mm_extract_epi64(mixed, 1)) as u64 | 1
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
        if at >= self.len {
            return None;
        }
        let at_line = at == 0 || *self.text.add(at - 1) == b'\n';
        let mut line = if at_line { at } else { past_line(at) };
        if line >= self.len {
            return None;
        }
        let (mut id, mut digit) = (0, line);
        while *self.text.add(digit) != b'\t' {
            id = id * 10 + (*self.text.add(digit) - b'0') as usize;
            digit += 1;
        }
        while id % 64 != 0 {
            line = past_line(line);
            id += 1;
            if line >= self.len {
                return None;
            }
        }
        Some((line, id))
    }

    /// A token's action, with no branch on whether a query uses it.
    #[inline(always)]
    unsafe fn action_of(&self, hash: u64) -> u32 {
        let bucket = self.buckets.add(bucket_of(hash));
        let wanted = _mm256_set1_epi32(hash as i32);
        let same = _mm256_cmpeq_epi32(_mm256_load_si256(bucket.cast()), wanted);
        let found = _mm256_movemask_ps(_mm256_castsi256_ps(same)) as u32 | 1 << BUCKET_SLOTS;
        let action = (*bucket).actions[found.trailing_zeros() as usize];
        if action == OVERFLOW {
            return self.action_past(hash);
        }
        action
    }

    /// The action of a token whose bucket is full, so that its term may be in a later one.
    #[cold]
    unsafe fn action_past(&self, hash: u64) -> u32 {
        let mut index = bucket_of(hash);
        loop {
            let bucket = &*self.buckets.add(index);
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
    unsafe fn index_claim<const SAMPLE: bool>(
        &self,
        claim: usize,
        counts: &mut [u32],
        buffers: &mut Buffers,
    ) -> usize {
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

        let empty = std::ptr::null_mut();
        let mut cursors = [(empty, empty); SHARDS];
        let Buffers {
            starts,
            ends,
            lines,
            hashes,
            heavy_hits,
            list_hits,
            anchor_hits,
        } = buffers;
        let (mut token_count, mut end_count, mut line_count) = (0, 0, 0);
        let mut carry = 0u64; // whether the previous 32 bytes ended inside a token
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
                    lines[line_count] = (token_count + before.count_ones() as usize) as u32;
                    line_count += 1;
                    breaks &= breaks - 1;
                }
                // Unconditional stores, so a varying token count needs no loop to predict.
                let (found, finished) =
                    (opened.count_ones() as usize, closed.count_ones() as usize);
                for extra in 0..12 {
                    *starts.get_unchecked_mut(token_count + extra) =
                        (at + opened.trailing_zeros() as usize) as u32;
                    *ends.get_unchecked_mut(end_count + extra) =
                        (at + closed.trailing_zeros() as usize) as u32;
                    opened &= opened.wrapping_sub(1);
                    closed &= closed.wrapping_sub(1);
                }
                let mut extra = 12;
                while opened != 0 {
                    starts[token_count + extra] = (at + opened.trailing_zeros() as usize) as u32;
                    opened &= opened - 1;
                    extra += 1;
                }
                extra = 12;
                while closed != 0 {
                    ends[end_count + extra] = (at + closed.trailing_zeros() as usize) as u32;
                    closed &= closed - 1;
                    extra += 1;
                }
                token_count += found;
                end_count += finished;
                at += 64;
            }

            // Only whole documents go on, so that a phrase has all its tokens at hand.
            let ready = if line_count > 0 {
                lines[line_count - 1] as usize
            } else {
                0
            };
            let mut token = 0;
            if first_token && ready > 0 {
                first_token = false;
                token = 1; // the id of the claim's first document
                hashes[0] = 0;
            }
            let mut i = token;
            while i + 1 < ready {
                let (from_a, to_a) = (
                    *starts.get_unchecked(i) as usize,
                    *ends.get_unchecked(i) as usize,
                );
                let (from_b, to_b) = (
                    *starts.get_unchecked(i + 1) as usize,
                    *ends.get_unchecked(i + 1) as usize,
                );
                let (len_a, len_b) = (to_a - from_a, to_b - from_b);
                if (len_a | len_b) > 16 {
                    *hashes.get_unchecked_mut(i) = hash_of(text.add(from_a), len_a);
                    *hashes.get_unchecked_mut(i + 1) = hash_of(text.add(from_b), len_b);
                } else {
                    let masks = MASKS.0.as_ptr();
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
                    *hashes.get_unchecked_mut(i) = _mm256_extract_epi64::<0>(folded) as u64 | 1;
                    *hashes.get_unchecked_mut(i + 1) = _mm256_extract_epi64::<2>(folded) as u64 | 1;
                }
                i += 2;
            }
            if i < ready {
                let (from, to) = (
                    *starts.get_unchecked(i) as usize,
                    *ends.get_unchecked(i) as usize,
                );
                *hashes.get_unchecked_mut(i) = hash_of(text.add(from), to - from);
            }
            let (mut heavy_count, mut list_count, mut anchor_count) = (0, 0, 0);
            for line in 0..line_count {
                let to = lines[line] as usize;
                while token < to {
                    let ahead = bucket_of(*hashes.get_unchecked(token + AHEAD));
                    _mm_prefetch::<_MM_HINT_T0>(self.buckets.add(ahead).cast());
                    let action = self.action_of(*hashes.get_unchecked(token));
                    // All three stores happen, and only the right counts move.
                    *heavy_hits.get_unchecked_mut(heavy_count) = action << DOC_BITS | doc as u32;
                    heavy_count += (action >> 31) as usize;
                    *list_hits.get_unchecked_mut(list_count) = (action as u64) << 32 | doc as u64;
                    list_count += (action.wrapping_sub(1) < HEAVY - 1) as usize;
                    // Rare enough to predict: about 1 token in 150 starts a phrase check.
                    if action & ANCHOR != 0 {
                        *anchor_hits.get_unchecked_mut(anchor_count) =
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
                    hashes[to] = 0;
                    token = to + 1;
                }
            }
            for (i, &hit) in anchor_hits[..anchor_count].iter().enumerate() {
                let later = |ahead: usize| (anchor_hits[i + ahead] >> 40) as usize;
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
                        && hashes[token - pivot..token - pivot + len]
                            == self.merged().words[first..first + len]
                    {
                        list_hits[list_count] =
                            (phrase.term as u64 + 1) << 32 | (hit >> 20 & ((1 << 20) - 1));
                        list_count += 1;
                    }
                }
            }
            if heavy_count > 0 {
                let phase = claim / PHASE_CLAIMS;
                let (set, base) = (self.bitmap(0, phase), self.phase_base[phase]);
                let width = self.words_per_bitmap;
                for &hit in &heavy_hits[..heavy_count] {
                    let doc = (hit & ((1 << DOC_BITS) - 1)) - base;
                    let at = (hit >> DOC_BITS) as usize * width + (doc / 64) as usize;
                    *set.add(at) |= 1 << (doc % 64);
                }
            }
            for &hit in &list_hits[..list_count] {
                let term = ((hit >> 32) as u32 & !ANCHOR) - 1;
                if SAMPLE {
                    counts[term as usize] += 1;
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
                lines[0] = 0;
                line_count = 1;
            }
            starts.copy_within(ready..token_count, 0);
            token_count -= ready;
            ends.copy_within(ready..end_count, 0);
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
        // The sample's rows do not exist, so its claims are read again by the main pass.
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
        let counts = self.sample_counts();
        let merged = self.merged();
        for index in
            merged.phrase_count * slice / PLAN_JOBS..merged.phrase_count * (slice + 1) / PLAN_JOBS
        {
            let phrase = &mut *self.phrases.add(index);
            let first = phrase.first as usize;
            let terms = &merged.word_terms[first..first + phrase.len as usize];
            let pivot = (0..terms.len())
                .min_by_key(|&i| counts.0[terms[i] as usize])
                .unwrap();
            let anchor = &mut (*self.terms.add(terms[pivot] as usize)).phrases;
            phrase.pivot = pivot as u32;
            phrase.next = AtomicU32::from_ptr(anchor).swap(index as u32 + 1, Relaxed);
        }
    }

    /// The sample's counts and document count. They are final by the time anyone asks, so no
    /// lock is held while they are read.
    unsafe fn sample_counts(&self) -> (&[u32], usize) {
        let samples = self.samples.lock().unwrap();
        (&*(samples.0.as_slice() as *const [u32]), samples.1)
    }

    /// Gives the actions in a slice of the buckets their final form, once the phrases have been
    /// planned. Slices are taken by whichever thread is free.
    unsafe fn plan_buckets(&self, slice: usize) {
        let (counts, docs) = self.sample_counts();
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
                } else if counts[id] as usize * HEAVY_SHARE >= docs {
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
        for low in 0..terms {
            if let Some(info) = (low * SHARDS + shard < term_count)
                .then(|| &mut *self.terms.add(low * SHARDS + shard))
                .filter(|info| !info.heavy)
            {
                info.first = 0;
                info.count = 0;
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
        let steps_of =
            |query: &Query| &share.steps[query.first as usize..(query.first + query.len) as usize];
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
        add_queries(self, &hash_queries(self.query_text, start(k), start(k + 1)))
    }

    /// Lists the shares' work units and joins the words of phrases, which are few. Runs on one
    /// thread while the others wait.
    unsafe fn merge(&self) {
        let mut all = Merged {
            shares: Vec::new(),
            units: Vec::new(),
            words: Vec::new(),
            word_terms: Vec::new(),
            phrase_count: 0,
        };
        for (index, share) in self.parsed.iter().enumerate() {
            let mut share = std::mem::replace(&mut *share.lock().unwrap(), Parsed::empty());
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
        // With no documents there is no phase to write the answers, and every answer is empty.
        let phases = self.phase_base.len() - 1;
        *self.parts.lock().unwrap() = all
            .units
            .iter()
            .map(|&(share, first)| match phases {
                0 => {
                    let left = all.shares[share as usize].queries.len() - first as usize;
                    b"0 0
"
                    .repeat(QUERY_BATCH.min(left))
                }
                _ => Vec::new(),
            })
            .collect();
        *self.samples.lock().unwrap() = (vec![0; self.term_count.load(Relaxed)], 0);
        self.merged.set(all).ok();
    }

    unsafe fn work(&self, id: usize) {
        // One mapping for all of a thread's buffers: each mapping call stalls the page faults
        // of the other threads.
        let mut region = reserve(
            (MAX_TOKENS + 64) * (4 + 4 + 8 + 4 + 8 + 8) + BATCH * 4,
            MADV_NOHUGEPAGE,
        );
        let mut carve = |count: usize, size: usize| {
            let part = region;
            region = region.add(count * size);
            part
        };
        let tokens = MAX_TOKENS + 32;
        let mut buffers = Buffers {
            starts: std::slice::from_raw_parts_mut(carve(tokens, 4).cast(), tokens),
            ends: std::slice::from_raw_parts_mut(carve(tokens, 4).cast(), tokens),
            lines: std::slice::from_raw_parts_mut(carve(BATCH / 2 + 64, 4).cast(), BATCH / 2 + 64),
            hashes: std::slice::from_raw_parts_mut(carve(tokens + AHEAD, 8).cast(), tokens + AHEAD),
            heavy_hits: std::slice::from_raw_parts_mut(carve(tokens, 4).cast(), tokens),
            list_hits: std::slice::from_raw_parts_mut(carve(tokens, 8).cast(), tokens),
            anchor_hits: std::slice::from_raw_parts_mut(carve(tokens, 8).cast(), tokens),
        };
        *self.parsed[id].lock().unwrap() = self.parse_share(id);
        if self.step.wait().is_leader() {
            self.merge();
        }
        self.step.wait();
        let mut counts = vec![0u32; self.term_count.load(Relaxed)];
        // Threads that wake late find fewer claims left, instead of delaying everyone.
        let mut docs = 0;
        loop {
            let claim = self.next_claim.fetch_add(1, Relaxed);
            if claim >= SAMPLE_CLAIMS {
                break;
            }
            docs += self.index_claim::<true>(claim, &mut counts, &mut buffers);
        }
        {
            let mut samples = self.samples.lock().unwrap();
            samples
                .0
                .iter_mut()
                .zip(&counts)
                .for_each(|(total, part)| *total += part);
            samples.1 += docs;
        }
        drop(counts);
        self.step.wait();
        self.each_job(&self.plan_jobs, PLAN_JOBS, |slice| self.plan_phrases(slice));
        self.step.wait();
        self.each_job(&self.bucket_jobs, PLAN_JOBS, |slice| {
            self.plan_buckets(slice)
        });
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
                let limit = (stage + 1) * PHASE_CLAIMS;
                loop {
                    let claim =
                        stage * PHASE_CLAIMS + self.phase_claims[stage].fetch_add(1, Relaxed);
                    if claim >= limit || claim * CLAIM >= self.len {
                        break;
                    }
                    self.index_claim::<false>(claim, &mut [], &mut buffers);
                }
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

fn holds(bits: &[u64], doc: u16) -> bool {
    unsafe { *bits.get_unchecked(doc as usize / 64) >> (doc % 64) & 1 == 1 }
}

/// One step of a BOOL query. The result goes in the scratch list the current set is not in.
unsafe fn apply(op: Op, current: Set, term: Set, scratch: &mut Scratch) -> Set {
    let Scratch {
        lists,
        active,
        bits,
        marks,
    } = scratch;
    let words = bits.len();
    *active ^= 1;
    let out = &mut lists[*active];
    out.clear();
    let own = |bits: &mut Vec<u64>, set: Set| {
        let target = bits.as_mut_ptr();
        match set {
            Set::Bits(source) => {
                if source.as_ptr() != target as *const u64 {
                    std::ptr::copy_nonoverlapping(source.as_ptr(), target, words);
                }
            }
            Set::List(list) => {
                bits.fill(0);
                for &doc in list {
                    *target.add(doc as usize / 64) |= 1 << (doc % 64);
                }
            }
        }
    };
    let finish_list =
        |out: &Vec<u16>| Set::List(std::slice::from_raw_parts(out.as_ptr(), out.len()));
    let finish_bits = |bits: &Vec<u64>| Set::Bits(std::slice::from_raw_parts(bits.as_ptr(), words));
    match (op, current, term) {
        (Op::And, Set::List(a), Set::List(b)) => {
            if a.len() <= b.len() {
                filter_by(a, b, true, out, marks);
            } else {
                filter_by(b, a, true, out, marks);
            }
            finish_list(out)
        }
        (Op::And | Op::Not, Set::List(a), Set::Bits(b)) | (Op::And, Set::Bits(b), Set::List(a)) => {
            out.reserve(a.len());
            // Keep what the bitmap holds for AND, and what it lacks for NOT.
            out.extend(a.iter().filter(|&&doc| holds(b, doc) == (op == Op::And)));
            finish_list(out)
        }
        (Op::Not, Set::List(a), Set::List(b)) => {
            filter_by(a, b, false, out, marks);
            finish_list(out)
        }
        (Op::Or, Set::List(a), Set::List(b)) if a.len() + b.len() < words => {
            out.reserve(a.len() + b.len());
            union(a, b, out);
            finish_list(out)
        }
        (Op::Not | Op::Or, a, Set::List(b)) | (Op::Or, Set::List(b), a) => {
            own(bits, a);
            let target = bits.as_mut_ptr();
            for &doc in b {
                let bit = 1 << (doc % 64);
                let word = &mut *target.add(doc as usize / 64);
                *word = if op == Op::Or {
                    *word | bit
                } else {
                    *word & !bit
                };
            }
            finish_bits(bits)
        }
        (op, a, Set::Bits(b)) => {
            own(bits, a);
            for (word, &other) in bits.iter_mut().zip(b) {
                *word = match op {
                    Op::And => *word & other,
                    Op::Or => *word | other,
                    Op::Not => *word & !other,
                };
            }
            finish_bits(bits)
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

/// The words of a run of queries, hashed but not yet interned. `marks` holds, per word, its
/// operator, and flags for the last word of a query and for a phrase.
struct Hashed {
    hashes: Vec<u64>,
    marks: Vec<u8>,
}

const LAST_WORD: u8 = 4;
const IN_PHRASE: u8 = 8;

/// Hashes every word of the queries in `bytes[from..to]`, which start and end at lines.
unsafe fn hash_queries(text: *const u8, from: usize, to: usize) -> Hashed {
    let mut found = Hashed {
        hashes: Vec::with_capacity((to - from) / 6 + 8),
        marks: Vec::with_capacity((to - from) / 6 + 8),
    };
    let mut at = from;
    while at < to {
        let phrase = *text.add(at) == b'P';
        at += if phrase { 8 } else { 5 }; // past `PHRASE "` or `BOOL `
        let mut op = Op::And;
        loop {
            let word = at;
            // Spaces, quotes and newlines all sort below the term bytes.
            while *text.add(at) >= b'0' {
                at += 1;
            }
            found.hashes.push(hash_of(text.add(word), at - word));
            let stop = *text.add(at);
            at += 1;
            let last = stop != b' ';
            found
                .marks
                .push(op as u8 | (last as u8) * LAST_WORD | (phrase as u8) * IN_PHRASE);
            if last {
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
    }
    found
}

impl Parsed {
    fn empty() -> Parsed {
        Parsed {
            queries: Vec::new(),
            steps: Vec::new(),
            tallies: &mut [],
            words: Vec::new(),
            word_terms: Vec::new(),
            phrases: Vec::new(),
        }
    }
}

/// One thread's share of the queries, with offsets that start at zero in each share.
struct Parsed {
    queries: Vec<Query>,
    steps: Vec<(Op, u32)>,
    tallies: &'static mut [Tally],
    words: Vec<u64>,
    word_terms: Vec<u32>,
    phrases: Vec<Phrase>,
}

/// Interns the hashed words in query order. The table is one cache line per bucket and mostly
/// cold, so each bucket is prefetched a few words ahead.
unsafe fn add_queries(run: &Run, found: &Hashed) -> Parsed {
    let mut parsed = Parsed::empty();
    let mut ids = (0, 0);
    let (mut first, mut first_word) = (0u32, 0u32);
    for (at, (&hash, &mark)) in found.hashes.iter().zip(&found.marks).enumerate() {
        if let Some(&later) = found.hashes.get(at + 8) {
            _mm_prefetch::<_MM_HINT_T0>(run.buckets.add(bucket_of(later)).cast());
        }
        let term = intern(run, &mut ids, hash);
        if mark & IN_PHRASE != 0 {
            parsed.words.push(hash);
            parsed.word_terms.push(term);
        } else {
            (*run.terms.add(term as usize)).in_bool = true;
            let op = [Op::And, Op::Or, Op::Not][(mark & 3) as usize];
            parsed.steps.push((op, term));
        }
        if mark & LAST_WORD == 0 {
            continue;
        }
        if mark & IN_PHRASE != 0 {
            let term = new_id(run, &mut ids);
            parsed.steps.push((Op::And, term));
            parsed.phrases.push(Phrase {
                term,
                first: first_word,
                len: parsed.words.len() as u32 - first_word,
                pivot: 0,
                next: 0,
            });
        }
        let len = parsed.steps.len() as u32 - first;
        parsed.queries.push(Query { first, len });
        (first, first_word) = (parsed.steps.len() as u32, parsed.words.len() as u32);
    }
    parsed.tallies = space(parsed.queries.len());
    parsed
}

/// Everything the threads parsed. A share keeps its own queries, steps and tallies; the
/// answers go through `units`, runs of up to QUERY_BATCH queries of one share, in file order.
struct Merged {
    shares: Vec<Parsed>,
    units: Vec<(u32, u32)>,
    words: Vec<u64>,
    word_terms: Vec<u32>,
    phrase_count: usize,
}

fn spawn_tree<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    run: &'env Run,
    id: usize,
    threads: usize,
) {
    // Each thread starts its children before doing its own work, so that 12 threads are
    // running after a few rounds of spawning, not after twelve.
    for child in [2 * id + 1, 2 * id + 2]
        .into_iter()
        .filter(|&c| c < threads)
    {
        let builder = thread::Builder::new().stack_size(1 << 20);
        builder
            .spawn_scoped(scope, move || spawn_tree(scope, run, child, threads))
            .unwrap();
    }
    unsafe { run.work(id) };
}

fn main() {
    let file = File::open("/data/docs.txt").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let queries = File::open("/data/queries.txt").unwrap();
    let query_len = queries.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    // One counter per phase, each starting at `init`.
    let counters = |init| {
        (0..len.div_ceil(PHASE_CLAIMS * CLAIM))
            .map(|_| AtomicUsize::new(init))
            .collect()
    };
    let mut run = unsafe {
        let text = map_file(&file, len);
        // The last line's id gives the document count.
        let body = std::slice::from_raw_parts(text, len);
        let last_end = len.saturating_sub(1);
        let last_line = body[..last_end]
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(0, |at| at + 1);
        let digits = body[last_line..]
            .iter()
            .position(|&c| c == b'\t')
            .unwrap_or(0);
        let last_id = body[last_line..last_line + digits]
            .iter()
            .fold(0, |v, &d| v * 10 + (d - b'0') as usize);
        let docs = if len == 0 { 0 } else { last_id + 1 };
        assert!(docs < 1 << DOC_BITS);
        let stride = len + 4096; // rows per shard: more than any shard can hold
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
            parsed: (0..threads).map(|_| Mutex::new(Parsed::empty())).collect(),
            merged: OnceLock::new(),
            phrases: reserve(MAX_TERMS * size_of::<Phrase>(), MADV_NOHUGEPAGE).cast(),
            samples: Mutex::new((Vec::new(), 0)),
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
            bitmaps: std::ptr::null_mut(),
            rows: reserve(SHARDS * stride * 4, MADV_NOHUGEPAGE).cast(),
            stride,
            claimed: (0..SHARDS).map(|_| AtomicUsize::new(0)).collect(),
            postings: reserve(len * 2 + HUGE_PAGE, MADV_HUGEPAGE).cast(),
            parts: Mutex::new(Vec::new()),
        }
    };
    // Phases start on documents that are multiples of 64, and a bitmap covers one phase.
    let phases = len.div_ceil(PHASE_CLAIMS * CLAIM);
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
    run.words_per_bitmap = widest as usize / 64 + 2;
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

    thread::scope(|s| spawn_tree(s, &run, 0, threads));

    let mut out = File::create("/out/results.txt").unwrap();
    for part in run.parts.lock().unwrap().iter() {
        out.write_all(part).unwrap();
    }
}

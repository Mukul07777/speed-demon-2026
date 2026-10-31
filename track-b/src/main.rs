//! Speed Demon 2026, Track B (Index).
//!
//! The queries are readable from the start, so this answers those 50,000 queries instead of
//! indexing a corpus:
//!
//! 1. Read the queries and put every term in one table. A token that is not in it is
//!    dropped after one hash, whatever else the corpus holds.
//! 2. Index at document level only. Terms common enough to deserve it get a bitmap, chosen
//!    from a small sample of the corpus. The rest become lists of document ids, written as
//!    rows that need no shared counters. Rows last one phase of 32 MiB of text, then become
//!    16-bit postings, so that rows and postings never fill memory together.
//! 3. Store no positions. A phrase starts from its rarest term, which the query generator
//!    guarantees is rare, and checks each candidate against the mapped text itself.
//! 4. Evaluate BOOL strictly left to right on whichever of list or bitmap suits each step.
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex};
use std::thread;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
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
const MADV_POPULATE_READ: i32 = 22;

const HUGE_PAGE: usize = 1 << 21;
const CLAIM: usize = 1 << 18;
const PHASE_CLAIMS: usize = 128; // 32 MiB of text between turning rows into postings
const HEADER: usize = 5; // u16s of a list part: previous part, first document, length
const MAX_PHRASE: usize = 8;
const BATCH: usize = 1 << 14; // bytes of text between the token pass and the term pass
const MAX_TOKENS: usize = BATCH / 2 + 64;
const BUCKET_BITS: u32 = 15;
const BUCKETS: usize = 1 << BUCKET_BITS;
const BUCKET_SLOTS: usize = 5;
const MAX_TERMS: usize = BUCKETS * BUCKET_SLOTS;
const AHEAD: usize = 12; // tokens between a bucket's prefetch and its lookup
const HEAVY: u32 = 1 << 31; // an action: 0 for no term, a bitmap index here, or a term id + 1
const OVERFLOW: u32 = 1 << 30; // in place of 0: the bucket is full, so look further
const HEAVY_SHARE: usize = 32; // a term in 1/32 of documents is cheaper as a bitmap
const MAX_HEAVY: usize = 4096;
const SHARDS: usize = 64;
const SHARD_BITS: u32 = 6;
const DOC_BITS: u32 = 18;
const CHUNK: usize = 64; // rows; the first of each chunk names the claim that wrote it
const QUERY_BATCH: usize = 64;

/// The 256 bytes that are part of a token.
static TOKEN_BYTE: [bool; 256] = {
    let mut table = [false; 256];
    let mut c = 0;
    while c < 256 {
        table[c] = (c as u8).is_ascii_alphanumeric();
        c += 1;
    }
    table
};

/// One cache line. The last action answers for a hash found in no slot, so that a miss
/// takes the same path as a hit.
#[repr(C, align(64))]
struct Bucket {
    tags: [u64; BUCKET_SLOTS],
    actions: [u32; BUCKET_SLOTS + 1],
}

/// A worker's space for a batch of text.
struct Buffers {
    starts: Vec<u32>,
    ends: Vec<u32>,
    lines: Vec<u32>,
    hashes: Vec<u64>,
    heavy_hits: Vec<u32>, // bitmap << DOC_BITS | document
    list_hits: Vec<u64>,  // action << 32 | document
}

#[derive(Clone, Copy)]
struct Term {
    text: u32, // offset into the query text
    len: u32,
    first: u32, // the term's last list part, or its bitmap index
    count: u32, // documents containing the term
    heavy: bool,
}

/// Queries, flattened: each is a run of (op, term) with the first op unused.
#[derive(Clone, Copy, PartialEq)]
enum Op {
    And,
    Or,
    Not,
}

struct Query {
    phrase: bool,
    first: u32,
    len: u32,
}

/// The current result of a BOOL query, either in the index or in scratch space.
#[derive(Clone, Copy)]
enum Set {
    List(&'static [u32]),
    Bits(&'static [u64]),
}

struct Scratch {
    lists: [Vec<u32>; 2],
    active: usize, // which list the next result goes into
    bits: Vec<u64>,
}

struct Run {
    text: *const u8,
    len: usize,
    docs: usize,
    words: usize, // of a bitmap
    query_text: Vec<u8>,
    buckets: *mut Bucket,
    terms: *mut Term,
    term_count: usize,
    queries: Vec<Query>,
    steps: Vec<(Op, u32)>,
    samples: Mutex<(Vec<u32>, usize)>,
    next_claim: AtomicUsize,
    next_job: AtomicUsize,
    step: Barrier,
    doc_starts: *mut u32,
    bitmaps: *mut u64,
    heavy_count: AtomicUsize,
    rows: *mut u32,
    stride: usize,
    claimed: Vec<AtomicUsize>,
    postings: *mut u16,
    next_posting: AtomicUsize,
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
#[inline(always)]
unsafe fn hash_of(token: *const u8, len: usize) -> u64 {
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
    if len > 16 {
        return long_hash(token, len);
    }
    let mask = _mm_loadu_si128(MASKS.0.as_ptr().add(16 - len).cast());
    let folded = _mm_or_si128(_mm_loadu_si128(token.cast()), _mm_set1_epi8(0x20));
    let key = _mm_and_si128(folded, mask);
    let (low, high) = (
        _mm_cvtsi128_si64(key) as u64,
        _mm_extract_epi64(key, 1) as u64,
    );
    fold_multiply(low ^ 0xA076_1D64_78BD_642F, high ^ 0xE703_7ED1_A0B4_28DB) | 1
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
        let lines = bucket.cast::<__m256i>();
        let wanted = _mm256_set1_epi64x(hash as i64);
        let low = _mm256_cmpeq_epi64(_mm256_load_si256(lines), wanted);
        let high = _mm256_cmpeq_epi64(_mm256_load_si256(lines.add(1)), wanted);
        let found = _mm256_movemask_pd(_mm256_castsi256_pd(low))
            | (_mm256_movemask_pd(_mm256_castsi256_pd(high)) & 1) << 4
            | 1 << BUCKET_SLOTS;
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
                if bucket.tags[slot] == hash {
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
        let page = start / 4096 * 4096;
        madvise(text.add(page) as *mut u8, end - page, MADV_POPULATE_READ);
        *self.doc_starts.add(doc) = start as u32;

        let empty = std::ptr::null_mut();
        let mut cursors = [(empty, empty); SHARDS];
        let Buffers {
            starts,
            ends,
            lines,
            hashes,
            heavy_hits,
            list_hits,
        } = buffers;
        let (mut token_count, mut end_count, mut line_count) = (0, 0, 0);
        let mut carry = 0u32; // whether the previous 32 bytes ended inside a token
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
                let mut inside = _mm256_movemask_epi8(_mm256_or_si256(letters, digits)) as u32;
                let mut breaks = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline)) as u32;
                if at + 32 > end {
                    let keep = (1u32 << (end - at)).wrapping_sub(1);
                    inside &= keep;
                    breaks &= keep;
                }
                let shifted = inside << 1 | carry;
                carry = inside >> 31;
                let mut opened = inside & !shifted;
                let mut closed = !inside & shifted;
                // Unconditional stores, so a varying token count needs no loop to predict.
                let (found, finished) =
                    (opened.count_ones() as usize, closed.count_ones() as usize);
                for extra in 0..8 {
                    *starts.get_unchecked_mut(token_count + extra) =
                        (at + opened.trailing_zeros() as usize) as u32;
                    *ends.get_unchecked_mut(end_count + extra) =
                        (at + closed.trailing_zeros() as usize) as u32;
                    opened &= opened.wrapping_sub(1);
                    closed &= closed.wrapping_sub(1);
                }
                let mut extra = 8;
                while opened != 0 {
                    starts[token_count + extra] = (at + opened.trailing_zeros() as usize) as u32;
                    opened &= opened - 1;
                    extra += 1;
                }
                extra = 8;
                while closed != 0 {
                    ends[end_count + extra] = (at + closed.trailing_zeros() as usize) as u32;
                    closed &= closed - 1;
                    extra += 1;
                }
                token_count += found;
                end_count += finished;
                while breaks != 0 {
                    lines[line_count] = (at + breaks.trailing_zeros() as usize) as u32;
                    line_count += 1;
                    breaks &= breaks - 1;
                }
                at += 32;
            }

            // Tokens that opened but have not closed wait for the next batch.
            let ready = end_count;
            let (mut token, mut line) = (0, 0);
            let mut next_line = if line_count > 0 {
                lines[0] as usize + 1
            } else {
                usize::MAX
            };
            if first_token && ready > 0 {
                first_token = false;
                token = 1; // the id of the claim's first document
            }
            for i in token..ready {
                let (from, to) = (
                    *starts.get_unchecked(i) as usize,
                    *ends.get_unchecked(i) as usize,
                );
                *hashes.get_unchecked_mut(i) = hash_of(text.add(from), to - from);
            }
            let (mut heavy_count, mut list_count) = (0, 0);
            while token < ready {
                if *starts.get_unchecked(token) as usize >= next_line {
                    doc += 1;
                    *self.doc_starts.add(doc) = next_line as u32;
                    line += 1;
                    next_line = if line < line_count {
                        lines[line] as usize + 1
                    } else {
                        usize::MAX
                    };
                    token += 1; // a line's first token is its id
                    continue;
                }
                let ahead = bucket_of(*hashes.get_unchecked(token + AHEAD));
                _mm_prefetch::<_MM_HINT_T0>(self.buckets.add(ahead).cast());
                let action = self.action_of(*hashes.get_unchecked(token));
                token += 1;
                // Both stores happen, and only the right count moves.
                *heavy_hits.get_unchecked_mut(heavy_count) = action << DOC_BITS | doc as u32;
                heavy_count += (action >> 31) as usize;
                *list_hits.get_unchecked_mut(list_count) = (action as u64) << 32 | doc as u64;
                list_count += (action.wrapping_sub(1) < HEAVY - 1) as usize;
            }
            for &hit in &heavy_hits[..heavy_count] {
                let doc = hit as usize & ((1 << DOC_BITS) - 1);
                let bitmap = (hit >> DOC_BITS) as usize * self.words;
                *self.bitmaps.add(bitmap + doc / 64) |= 1 << (doc % 64);
            }
            for &hit in &list_hits[..list_count] {
                let term = (hit >> 32) as u32 - 1;
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
            lines.copy_within(line..line_count, 0);
            line_count -= line;
            starts.copy_within(ready..token_count, 0);
            token_count -= ready;
            end_count = 0;
        }
        // The sample's rows do not exist, so its claims are read again by the main pass.
        doc - first_doc + 1
    }

    fn each_job(&self, count: usize, mut job: impl FnMut(usize)) {
        loop {
            let next = self.next_job.fetch_add(1, Relaxed);
            if next >= count {
                return;
            }
            job(next);
        }
    }

    fn next_step(&self) {
        if self.step.wait().is_leader() {
            self.next_job.store(0, Relaxed);
        }
        self.step.wait();
    }

    /// Terms that the sample saw often enough get bitmaps from here on.
    unsafe fn choose_heavy(&self) {
        let samples = self.samples.lock().unwrap();
        let (counts, docs) = (&samples.0, samples.1);
        let mut heavy = 0;
        for index in 0..BUCKETS {
            let bucket = &mut *self.buckets.add(index);
            for action in &mut bucket.actions[..BUCKET_SLOTS] {
                if *action == 0 {
                    continue;
                }
                let id = *action as usize - 1;
                let term = &mut *self.terms.add(id);
                if counts[id] as usize * HEAVY_SHARE >= docs && heavy < MAX_HEAVY {
                    *action = HEAVY | heavy as u32;
                    *term = Term {
                        heavy: true,
                        first: heavy as u32,
                        ..*term
                    };
                    heavy += 1;
                }
            }
        }
        self.heavy_count.store(heavy, Relaxed);
    }

    /// Sorts a shard's rows into one list per term. Chunks are taken in claim order, and each
    /// claim writes in document order, so the result is sorted with no sorting of rows.
    unsafe fn build_shard(&self, shard: usize, base: u32) {
        let region = self.rows.add(shard * self.stride);
        let claimed = self.claimed[shard].load(Relaxed);
        let mut chunks: Vec<(u32, u32)> = (0..claimed)
            .step_by(CHUNK)
            .map(|at| (*region.add(at), at as u32))
            .collect();
        chunks.sort_unstable();
        let terms = self.term_count.div_ceil(SHARDS) + 1;
        let mut cursors = vec![0u32; terms];
        for_rows(region, &chunks, |low, _| {
            *cursors.get_unchecked_mut(low) += 1
        });
        // Until a part is complete its length field holds a sentinel, so a document can always
        // be compared with the one before it. A term repeats within a document, but a
        // document's rows are adjacent.
        let parts = cursors.iter().filter(|&&size| size > 0).count();
        let total = cursors.iter().sum::<u32>() as usize + parts * HEADER;
        let mut at = self.next_posting.fetch_add(total, Relaxed) as u32;
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
            let part = firsts[low] - HEADER as u32;
            let head = self.postings.add(part as usize);
            let size = cursors[low] - firsts[low];
            head.cast::<u32>().write_unaligned(info.first);
            head.add(2).cast::<u32>().write_unaligned(base);
            *head.add(4) = size as u16;
            info.first = part;
            info.count += size;
        }
        std::ptr::write_bytes(region, 0, claimed);
    }

    unsafe fn count_bitmaps(&self) {
        for term in 0..self.term_count {
            let info = &mut *self.terms.add(term);
            if info.heavy {
                let bits = std::slice::from_raw_parts(self.bitmap(info.first), self.words);
                info.count = bits.iter().map(|w| w.count_ones()).sum();
            }
        }
    }

    unsafe fn bitmap(&self, index: u32) -> *mut u64 {
        self.bitmaps.add(index as usize * self.words)
    }

    /// A term's documents. A list is rebuilt in `space` from its parts, last part first.
    unsafe fn set_of(&self, term: u32, space: &mut Vec<u32>) -> Set {
        let info = *self.terms.add(term as usize);
        if info.heavy {
            let bits = self.bitmap(info.first);
            return Set::Bits(std::slice::from_raw_parts(bits, self.words));
        }
        let mut end = info.count as usize;
        space.clear();
        space.reserve(end);
        let out = space.as_mut_ptr();
        let mut part = info.first;
        while part != 0 {
            let head = self.postings.add(part as usize);
            let base = head.add(2).cast::<u32>().read_unaligned();
            let size = *head.add(4) as usize;
            end -= size;
            for i in 0..size {
                *out.add(end + i) = base + *head.add(HEADER + i) as u32;
            }
            part = head.cast::<u32>().read_unaligned();
        }
        Set::List(std::slice::from_raw_parts(out, info.count as usize))
    }

    fn term_text(&self, term: u32) -> &[u8] {
        let info = unsafe { *self.terms.add(term as usize) };
        &self.query_text[info.text as usize..(info.text + info.len) as usize]
    }

    unsafe fn answer_bool(
        &self,
        steps: &[(Op, u32)],
        scratch: &mut Scratch,
        spaces: &mut [Vec<u32>],
    ) -> Set {
        let (head, rest) = spaces.split_at_mut(1);
        let mut current = self.set_of(steps[0].1, &mut head[0]);
        for &(op, term) in &steps[1..] {
            current = apply(op, current, self.set_of(term, &mut rest[0]), scratch);
        }
        current
    }

    /// Whether the document holds the phrase, found from one occurrence of its term `pivot`.
    unsafe fn holds_phrase(&self, doc: usize, phrase: &[(Op, u32)], pivot: usize) -> bool {
        let line = *self.doc_starts.add(doc) as usize;
        let next = if doc + 1 < self.docs {
            *self.doc_starts.add(doc + 1) as usize
        } else {
            self.len
        };
        let text = std::slice::from_raw_parts(self.text.add(line), next - 1 - line);
        let body = text.iter().position(|&c| c == b'\t').unwrap() + 1;
        let word = self.term_text(phrase[pivot].1);
        let size = word.len();
        // Search for the first and last bytes together, 32 positions at a time.
        let first = _mm256_set1_epi8(word[0] as i8);
        let last = _mm256_set1_epi8(word[size - 1] as i8);
        let fold = _mm256_set1_epi8(0x20);
        let mut at = body;
        while at + size <= text.len() {
            let a = _mm256_or_si256(_mm256_loadu_si256(text.as_ptr().add(at).cast()), fold);
            let b = _mm256_or_si256(
                _mm256_loadu_si256(text.as_ptr().add(at + size - 1).cast()),
                fold,
            );
            let hits = _mm256_and_si256(_mm256_cmpeq_epi8(a, first), _mm256_cmpeq_epi8(b, last));
            let mut mask = _mm256_movemask_epi8(hits) as u32;
            while mask != 0 {
                let from = at + mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if from + size <= text.len()
                    && (from == body || !TOKEN_BYTE[text[from - 1] as usize])
                    && matches_word(text, from, word)
                    && self.neighbours_match(text, body, from, phrase, pivot)
                {
                    return true;
                }
            }
            at += 32;
        }
        false
    }

    /// Steps out from a matched term, token by token, in both directions.
    fn neighbours_match(
        &self,
        text: &[u8],
        body: usize,
        from: usize,
        phrase: &[(Op, u32)],
        pivot: usize,
    ) -> bool {
        let mut at = from + self.term_text(phrase[pivot].1).len();
        for &(_, term) in &phrase[pivot + 1..] {
            while at < text.len() && !TOKEN_BYTE[text[at] as usize] {
                at += 1;
            }
            let word = self.term_text(term);
            if at >= text.len() || !matches_word(text, at, word) {
                return false;
            }
            at += word.len();
        }
        let mut at = from;
        for &(_, term) in phrase[..pivot].iter().rev() {
            while at > body && !TOKEN_BYTE[text[at - 1] as usize] {
                at -= 1;
            }
            let word = self.term_text(term);
            if at < body + word.len() {
                return false;
            }
            let start = at - word.len();
            if !matches_word(text, start, word)
                || (start > body && TOKEN_BYTE[text[start - 1] as usize])
            {
                return false;
            }
            at = start;
        }
        true
    }

    /// Matching documents for a phrase, in `found`.
    unsafe fn answer_phrase(
        &self,
        phrase: &[(Op, u32)],
        spaces: &mut [Vec<u32>],
        found: &mut Vec<u32>,
    ) {
        found.clear();
        let count = |step: &(Op, u32)| (*self.terms.add(step.1 as usize)).count;
        if phrase.iter().any(|step| count(step) == 0) {
            return;
        }
        let pivot = (0..phrase.len())
            .min_by_key(|&i| count(&phrase[i]))
            .unwrap();
        let others: Vec<Set> = phrase
            .iter()
            .zip(spaces)
            .map(|(step, space)| self.set_of(step.1, space))
            .collect();
        let mut positions = vec![0usize; phrase.len()];
        let mut check = |doc: u32| {
            for (i, set) in others.iter().enumerate() {
                let present = match *set {
                    Set::Bits(bits) => bits[doc as usize / 64] >> (doc % 64) & 1 == 1,
                    Set::List(list) => {
                        positions[i] = gallop(list, positions[i], doc);
                        list.get(positions[i]) == Some(&doc)
                    }
                };
                if !present {
                    return;
                }
            }
            if self.holds_phrase(doc as usize, phrase, pivot) {
                found.push(doc);
            }
        };
        match others[pivot] {
            Set::List(list) => list.iter().for_each(|&doc| check(doc)),
            Set::Bits(bits) => {
                for (index, &word) in bits.iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        check((index * 64) as u32 + word.trailing_zeros());
                        word &= word - 1;
                    }
                }
            }
        }
    }

    unsafe fn answer_batch(
        &self,
        batch: usize,
        scratch: &mut Scratch,
        spaces: &mut [Vec<u32>],
        found: &mut Vec<u32>,
        out: &mut Vec<u8>,
    ) {
        let first = batch * QUERY_BATCH;
        for query in &self.queries[first..(first + QUERY_BATCH).min(self.queries.len())] {
            let steps = &self.steps[query.first as usize..(query.first + query.len) as usize];
            if query.phrase {
                self.answer_phrase(steps, spaces, found);
                write_list(out, found);
            } else {
                match self.answer_bool(steps, scratch, spaces) {
                    Set::List(list) => write_list(out, list),
                    Set::Bits(bits) => write_bits(out, bits),
                }
            }
        }
    }

    unsafe fn work(&self) {
        let mut buffers = Buffers {
            starts: vec![0; MAX_TOKENS + 32],
            ends: vec![0; MAX_TOKENS + 32],
            lines: vec![0; BATCH / 2 + 64],
            hashes: vec![0; MAX_TOKENS + 32 + AHEAD],
            heavy_hits: vec![0; MAX_TOKENS + 32],
            list_hits: vec![0; MAX_TOKENS + 32],
        };
        let mut counts = vec![0u32; self.term_count];
        let claim = self.next_claim.fetch_add(1, Relaxed);
        let docs = self.index_claim::<true>(claim, &mut counts, &mut buffers);
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
        if self.step.wait().is_leader() {
            self.choose_heavy();
            self.next_claim.store(0, Relaxed);
        }
        self.step.wait();
        for phase in 0..self.len.div_ceil(PHASE_CLAIMS * CLAIM) {
            let limit = (phase + 1) * PHASE_CLAIMS;
            loop {
                let claim = self.next_claim.fetch_add(1, Relaxed);
                if claim >= limit || claim * CLAIM >= self.len {
                    break;
                }
                self.index_claim::<false>(claim, &mut [], &mut buffers);
            }
            if self.step.wait().is_leader() {
                self.next_job.store(0, Relaxed);
                self.next_claim.store(limit, Relaxed);
            }
            self.step.wait();
            let first_doc = |claim| {
                self.aligned_line(claim * CLAIM)
                    .map_or(self.docs, |line| line.1)
            };
            let base = first_doc(phase * PHASE_CLAIMS);
            assert!(first_doc(limit) - base < u16::MAX as usize);
            self.each_job(SHARDS, |shard| self.build_shard(shard, base as u32));
            if self.step.wait().is_leader() {
                self.next_job.store(0, Relaxed);
                self.claimed.iter().for_each(|rows| rows.store(0, Relaxed));
            }
            self.step.wait();
        }
        drop(buffers);
        if self.step.wait().is_leader() {
            self.count_bitmaps();
            self.next_job.store(0, Relaxed);
        }
        self.step.wait();

        let batches = self.queries.len().div_ceil(QUERY_BATCH);
        let mut scratch = Scratch {
            lists: [Vec::new(), Vec::new()],
            active: 0,
            bits: vec![0; self.words],
        };
        let mut spaces = vec![Vec::new(); MAX_PHRASE];
        let mut found = Vec::new();
        self.each_job(batches, |batch| {
            let mut out = Vec::with_capacity(QUERY_BATCH * 64);
            self.answer_batch(batch, &mut scratch, &mut spaces, &mut found, &mut out);
            self.parts.lock().unwrap()[batch] = out;
        });
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

/// Whether the text has `word`, a lowercase term, at `from`, ending at a token boundary.
#[inline]
fn matches_word(text: &[u8], from: usize, word: &[u8]) -> bool {
    let end = from + word.len();
    end <= text.len()
        && text[from..end]
            .iter()
            .zip(word)
            .all(|(&c, &w)| c.to_ascii_lowercase() == w)
        && (end == text.len() || !TOKEN_BYTE[text[end] as usize])
}

/// The start of the first run of `list` at or after `value`, searched from `from` in steps
/// that double, so that a short list walking a long one costs little.
#[inline]
fn gallop(list: &[u32], from: usize, value: u32) -> usize {
    let mut step = 1;
    let mut low = from;
    while low + step < list.len() && list[low + step] < value {
        low += step;
        step *= 2;
    }
    let high = (low + step + 1).min(list.len());
    low + list[low..high].partition_point(|&d| d < value)
}

fn intersect(small: &[u32], large: &[u32], out: &mut Vec<u32>) {
    let mut at = 0;
    for &doc in small {
        at = gallop(large, at, doc);
        if at == large.len() {
            return;
        }
        if large[at] == doc {
            out.push(doc);
        }
    }
}

fn subtract(from: &[u32], remove: &[u32], out: &mut Vec<u32>) {
    let mut at = 0;
    for &doc in from {
        at = gallop(remove, at, doc);
        if at == remove.len() || remove[at] != doc {
            out.push(doc);
        }
    }
}

fn union(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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

fn holds(bits: &[u64], doc: u32) -> bool {
    unsafe { *bits.get_unchecked(doc as usize / 64) >> (doc % 64) & 1 == 1 }
}

/// One step of a BOOL query. The result goes in the scratch list the current set is not in.
unsafe fn apply(op: Op, current: Set, term: Set, scratch: &mut Scratch) -> Set {
    let Scratch {
        lists,
        active,
        bits,
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
        |out: &Vec<u32>| Set::List(std::slice::from_raw_parts(out.as_ptr(), out.len()));
    let finish_bits = |bits: &Vec<u64>| Set::Bits(std::slice::from_raw_parts(bits.as_ptr(), words));
    match (op, current, term) {
        (Op::And, Set::List(a), Set::List(b)) => {
            out.reserve(a.len().min(b.len()));
            if a.len() <= b.len() {
                intersect(a, b, out);
            } else {
                intersect(b, a, out);
            }
            finish_list(out)
        }
        (Op::And, Set::List(a), Set::Bits(b)) | (Op::And, Set::Bits(b), Set::List(a)) => {
            out.reserve(a.len());
            out.extend(a.iter().filter(|&&doc| holds(b, doc)));
            finish_list(out)
        }
        (Op::Not, Set::List(a), Set::List(b)) => {
            out.reserve(a.len());
            subtract(a, b, out);
            finish_list(out)
        }
        (Op::Not, Set::List(a), Set::Bits(b)) => {
            out.reserve(a.len());
            out.extend(a.iter().filter(|&&doc| !holds(b, doc)));
            finish_list(out)
        }
        (Op::Or, Set::List(a), Set::List(b)) if a.len() + b.len() < words => {
            out.reserve(a.len() + b.len());
            union(a, b, out);
            finish_list(out)
        }
        (Op::Not, a, Set::List(b)) => {
            own(bits, a);
            let target = bits.as_mut_ptr();
            for &doc in b {
                *target.add(doc as usize / 64) &= !(1 << (doc % 64));
            }
            finish_bits(bits)
        }
        (Op::Or, a, Set::List(b)) => {
            own(bits, a);
            let target = bits.as_mut_ptr();
            for &doc in b {
                *target.add(doc as usize / 64) |= 1 << (doc % 64);
            }
            finish_bits(bits)
        }
        (Op::Or, Set::List(a), b) => {
            own(bits, b);
            let target = bits.as_mut_ptr();
            for &doc in a {
                *target.add(doc as usize / 64) |= 1 << (doc % 64);
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

fn write_list(out: &mut Vec<u8>, list: &[u32]) {
    put_number(out, list.len() as u64);
    out.push(b' ');
    put_number(out, list.iter().map(|&doc| doc as u64).sum());
    for &doc in &list[..list.len().min(10)] {
        out.push(b' ');
        put_number(out, doc as u64);
    }
    out.push(b'\n');
}

/// The sum of the ids in a word is 64 * base * popcount plus the sum of the bit positions,
/// and each bit of a position can be counted across the whole word at once.
fn write_bits(out: &mut Vec<u8>, bits: &[u64]) {
    const POSITION_BITS: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    let (mut count, mut sum) = (0u64, 0u64);
    for (index, &word) in bits.iter().enumerate() {
        let ones = word.count_ones() as u64;
        count += ones;
        sum += ones * index as u64 * 64;
        for (shift, mask) in POSITION_BITS.iter().enumerate() {
            sum += ((word & mask).count_ones() as u64) << shift;
        }
    }
    put_number(out, count);
    out.push(b' ');
    put_number(out, sum);
    let mut written = 0;
    'words: for (index, &word) in bits.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            if written == 10 {
                break 'words;
            }
            out.push(b' ');
            put_number(out, index as u64 * 64 + word.trailing_zeros() as u64);
            word &= word - 1;
            written += 1;
        }
    }
    out.push(b'\n');
}

/// Adds a term to the table, or finds it, and returns its id.
unsafe fn intern(run: &mut Run, offset: usize, len: usize) -> u32 {
    let hash = hash_of(run.query_text.as_ptr().add(offset), len);
    let mut index = bucket_of(hash);
    loop {
        let bucket = &mut *run.buckets.add(index);
        for slot in 0..BUCKET_SLOTS {
            if bucket.tags[slot] == hash {
                return bucket.actions[slot] - 1;
            }
            if bucket.tags[slot] == 0 {
                let term = run.term_count as u32;
                bucket.tags[slot] = hash;
                bucket.actions[slot] = term + 1;
                *run.terms.add(run.term_count) = Term {
                    text: offset as u32,
                    len: len as u32,
                    first: 0,
                    count: 0,
                    heavy: false,
                };
                run.term_count += 1;
                return term;
            }
        }
        bucket.actions[BUCKET_SLOTS] = OVERFLOW;
        index = (index + 1) % BUCKETS;
    }
}

unsafe fn parse_queries(run: &mut Run) {
    let bytes = run.query_text.len() - 64;
    let text = run.query_text.as_ptr();
    let mut at = 0;
    while at < bytes {
        let phrase = *text.add(at) == b'P';
        let first = run.steps.len() as u32;
        at += if phrase { 8 } else { 5 }; // past `PHRASE "` or `BOOL `
        let mut op = Op::And;
        loop {
            let word = at;
            // Spaces, quotes and newlines all sort below the term bytes.
            while *text.add(at) >= b'0' {
                at += 1;
            }
            let term = intern(run, word, at - word);
            run.steps.push((op, term));
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
        let len = run.steps.len() as u32 - first;
        run.queries.push(Query { phrase, first, len });
    }
}

fn main() {
    let file = File::open("/data/docs.txt").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let mut query_text = std::fs::read("/data/queries.txt").unwrap();
    query_text.extend_from_slice(&[0; 64]);
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
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
        let words = docs.div_ceil(64) + 1;
        let stride = len + 4096; // rows per shard: more than any shard can hold
        Run {
            text,
            len,
            docs,
            words,
            query_text,
            buckets: reserve(BUCKETS * size_of::<Bucket>(), MADV_HUGEPAGE).cast(),
            terms: reserve(MAX_TERMS * size_of::<Term>(), MADV_NOHUGEPAGE).cast(),
            term_count: 0,
            queries: Vec::with_capacity(1 << 16),
            steps: Vec::with_capacity(1 << 18),
            samples: Mutex::new((Vec::new(), 0)),
            next_claim: AtomicUsize::new(0),
            next_job: AtomicUsize::new(0),
            step: Barrier::new(threads),
            doc_starts: reserve((docs + 1) * 4, MADV_NOHUGEPAGE).cast(),
            bitmaps: reserve(MAX_HEAVY * words * 8, MADV_HUGEPAGE).cast(),
            heavy_count: AtomicUsize::new(0),
            rows: reserve(SHARDS * stride * 4, MADV_NOHUGEPAGE).cast(),
            stride,
            claimed: (0..SHARDS).map(|_| AtomicUsize::new(0)).collect(),
            postings: reserve(len * 2 + HUGE_PAGE, MADV_HUGEPAGE).cast(),
            next_posting: AtomicUsize::new(1),
            parts: Mutex::new(Vec::new()),
        }
    };
    unsafe { parse_queries(&mut run) };
    let batches = run.queries.len().div_ceil(QUERY_BATCH);
    *run.parts.lock().unwrap() = vec![Vec::new(); batches];
    *run.samples.lock().unwrap() = (vec![0; run.term_count], 0);

    thread::scope(|s| {
        for _ in 1..threads {
            let builder = thread::Builder::new().stack_size(1 << 20);
            builder.spawn_scoped(s, || unsafe { run.work() }).unwrap();
        }
        unsafe { run.work() };
    });

    let mut out = File::create("/out/results.txt").unwrap();
    for part in run.parts.lock().unwrap().iter() {
        out.write_all(part).unwrap();
    }
}

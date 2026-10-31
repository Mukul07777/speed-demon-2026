//! Speed Demon 2026, Track C (Join).
//!
//! The file is parsed in parallel without ever scanning from its start, because the spec
//! gives a safe resync point: inside a note an LF is only ever followed by a lowercase word
//! or a quote, so an LF followed by a digit or a BOM always starts a record.
//!
//! Each worker claims 1 MiB and walks it in 16 KiB batches:
//!
//! 1. Find structure. For each 64 bytes, AVX2 finds quotes, commas and LFs, and a carry-less
//!    multiply turns the quotes into an inside-quotes mask. `""` toggles twice, so escapes
//!    need no handling at all. Every record then has exactly five structural bytes.
//! 2. Read fields. With no branch that depends on the data: 23% of records hold an escape,
//!    so branching on escapes or field shapes would mispredict constantly.
//! 3. Number the names. Names are drawn uniformly from 100,000 values, so every lookup is
//!    a cache miss, and the batch prefetches before it touches anything. A name is a 64-bit
//!    hash in a table of 64-byte buckets, so a lookup is one cache line and one AVX2 compare.
//!    Two different names sharing a hash would need odds of about 3 in 10 billion, well below
//!    the chance of a hardware fault, and the saving is a second memory access per record.
//!
//! Each record then sets its name's bit in its country's row of one shared bit matrix, and
//! popcounts of the rows give the distinct names.
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::ptr::read_unaligned;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex};
use std::thread;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, ...) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x4000;
const MADV_HUGEPAGE: i32 = 14;
const MADV_POPULATE_READ: i32 = 22;
const PR_SET_THP_DISABLE: i32 = 41;

const HUGE_PAGE: usize = 1 << 21;
const CLAIM: usize = 1 << 20;
const REGION_CLAIM: usize = 1 << 16;
const BATCH_WORDS: usize = 32; // 64-byte steps of the file between record passes
const FIELDS: usize = 5; // structural bytes per record: four commas and an LF
const MIN_RECORD: usize = 11; // `1,,,1,0.00` and an LF
const MAX_POSITIONS: usize = BATCH_WORDS * 64 / MIN_RECORD * FIELDS + 2 * FIELDS;
const MAX_RECORDS: usize = MAX_POSITIONS / FIELDS;
const QUOTE_WORDS: usize = 1 << 10; // masks kept for the rank query, more than one batch
const BUCKETS: usize = 1 << 15;
const BUCKET_SLOTS: usize = 5;
const MAX_NAMES: usize = 1 << 17; // the generator draws from 100,000
const REGION_IDS: usize = 1_000_000; // lookup ids stay below this; anything above is unmatched
const GROUPS: usize = 256; // 240 countries, with index 0 for unmatched
const SEEN_WORDS: usize = MAX_NAMES / 64; // of one group's row of the name bit matrix
const LETTERS: usize = 26 * 26 * 26;
const ZEROS: u64 = 0x3030_3030_3030_3030;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Stats {
    count: u64,
    id_sum: u64,
    amount_sum: i64,
    amount_min: i64,
    amount_max: i64,
    names_distinct: u64,
    name_bytes: u64,
    note_bytes: u64,
}

impl Stats {
    fn new() -> Stats {
        let (amount_min, amount_max) = (i64::MAX, i64::MIN);
        Stats {
            count: 0,
            id_sum: 0,
            amount_sum: 0,
            amount_min,
            amount_max,
            names_distinct: 0,
            name_bytes: 0,
            note_bytes: 0,
        }
    }

    fn merge(&mut self, other: &Stats) {
        self.count += other.count;
        self.id_sum += other.id_sum;
        self.amount_sum += other.amount_sum;
        self.amount_min = self.amount_min.min(other.amount_min);
        self.amount_max = self.amount_max.max(other.amount_max);
        self.names_distinct += other.names_distinct;
        self.name_bytes += other.name_bytes;
        self.note_bytes += other.note_bytes;
    }
}

/// One record between the passes of a batch.
#[derive(Clone, Copy)]
struct Pending {
    hash: u64,
    id: u64,
    amount: i64,
    region: u32,
    name_bytes: u32,
    note_bytes: u32,
}

/// One cache line: name hashes and their numbers + 1. Numbers are dense, which keeps the bit
/// matrix small.
#[repr(C, align(64))]
struct Bucket {
    tags: [AtomicU64; BUCKET_SLOTS],
    numbers: [AtomicU32; BUCKET_SLOTS],
}

struct Run {
    records: *const u8,
    records_len: usize,
    regions: *const u8,
    regions_len: usize,
    next_region_claim: AtomicUsize,
    next_claim: AtomicUsize,
    region_table: *mut u8,          // region id to group
    letters: Vec<AtomicU8>,         // country code to group
    countries: Mutex<Vec<[u8; 3]>>, // group - 1 to country code
    regions_done: Barrier,
    buckets: *mut Bucket,
    next_name: AtomicUsize,
    seen: *const AtomicU64, // SEEN_WORDS per group
    totals: Mutex<Vec<Stats>>,
}
unsafe impl Sync for Run {}

unsafe fn read(p: *const u8) -> u64 {
    read_unaligned(p.cast())
}

unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p.cast())
}

unsafe fn equal_mask(chunk: __m256i, byte: u8) -> u32 {
    _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(byte as i8))) as u32
}

/// The value of the eight digit bytes in `digits`, already minus '0', first digit lowest.
fn eight_digits(digits: u64) -> u64 {
    let v = (digits & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

/// The number written in the `len` bytes before `end`, for lengths 1 to 8. Shifting the
/// digits to the top makes the bytes below them read as leading zeros.
unsafe fn number_before(end: *const u8, len: usize) -> u64 {
    eight_digits((read(end.sub(8)) ^ ZEROS) >> (64 - 8 * len) << (64 - 8 * len))
}

/// An amount such as `-5.25` in the `len` bytes before `end`, in cents.
unsafe fn cents_before(end: *const u8, len: usize) -> i64 {
    let negative = (*end.sub(len) == b'-') as u64;
    let drop = 64 - 8 * len;
    let bytes = (read(end.sub(8)) ^ ZEROS) >> drop << drop;
    // The sign reads as a leading zero.
    let bytes = bytes & !((negative * 0xFF) << drop);
    // Close the gap the point leaves: its byte is always third from the end.
    let closed = (bytes & 0xFF_FFFF_FFFF) << 8 | bytes & 0xFFFF_0000_0000_0000;
    (eight_digits(closed) as i64 ^ -(negative as i64)) + negative as i64
}

/// Sets each bit that has an odd number of set bits at or below it, so quote bits become a
/// mask of everything inside quotes.
unsafe fn prefix_xor(bits: u64) -> u64 {
    let ones = _mm_set1_epi8(-1);
    _mm_cvtsi128_si64(_mm_clmulepi64_si128(
        _mm_cvtsi64_si128(bits as i64),
        ones,
        0,
    )) as u64
}

/// Multiplies to 128 bits and folds, the mixing step of wyhash.
#[inline(always)]
fn fold_multiply(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

/// The hash of a name, never 0, which marks an unused slot. Names are compared by their raw
/// bytes: every non-empty name is quoted, so equal raw bytes mean equal values and nothing
/// needs unescaping. Names of 32 bytes and more do not occur at seed 1 (the longest is 27),
/// so they cost one predicted branch here.
#[inline(always)]
unsafe fn name_hash(name: *const u8, len: usize) -> u64 {
    #[repr(align(32))]
    struct Masks([u8; 64]);
    static MASKS: Masks = {
        let mut bytes = [0u8; 64];
        let mut i = 0;
        while i < 32 {
            bytes[i] = 0xFF;
            i += 1;
        }
        Masks(bytes)
    };
    if len >= 32 {
        return long_name_hash(name, len);
    }
    let bytes = _mm256_and_si256(load(name), load(MASKS.0.as_ptr().add(32 - len)));
    let lane0 = _mm256_extract_epi64(bytes, 0) as u64;
    let lane1 = _mm256_extract_epi64(bytes, 1) as u64;
    let lane2 = _mm256_extract_epi64(bytes, 2) as u64;
    let lane3 = _mm256_extract_epi64(bytes, 3) as u64;
    let low = fold_multiply(lane0 ^ 0xA076_1D64_78BD_642F, lane1 ^ 0xE703_7ED1_A0B4_28DB);
    let high = fold_multiply(
        lane2 ^ 0x8EBC_6AF0_9C88_C6E3,
        lane3 ^ len as u64 ^ 0x5899_65CC_7537_4CC3,
    );
    fold_multiply(low ^ 0x1D8E_4E27_C47D_124F, high ^ 0xEB44_ACCA_B455_D165) | 1
}

#[cold]
unsafe fn long_name_hash(name: *const u8, len: usize) -> u64 {
    let bytes = std::slice::from_raw_parts(name, len);
    bytes.iter().fold(len as u64, |hash, &byte| {
        fold_multiply(hash ^ byte as u64, 0xE703_7ED1_A0B4_28DB)
    }) | 1
}

impl Run {
    /// Registers a country the first time it appears. Only 240 calls get past the first load.
    #[cold]
    fn group_of_country(&self, code: [u8; 3], cell: &AtomicU8) -> u8 {
        let mut countries = self.countries.lock().unwrap();
        if cell.load(Relaxed) == 0 {
            countries.push(code);
            assert!(countries.len() < GROUPS);
            cell.store(countries.len() as u8, Relaxed);
        }
        cell.load(Relaxed)
    }

    unsafe fn parse_regions(&self) {
        let input = self.regions;
        let len = self.regions_len;
        loop {
            let claim = self.next_region_claim.fetch_add(REGION_CLAIM, Relaxed);
            if claim >= len {
                return;
            }
            // Each claim starts after the first LF at or past its start, so the header goes
            // to nobody and a line belongs to the claim holding the LF before it.
            let seek = std::slice::from_raw_parts(input.add(claim), len - claim);
            let mut line = claim + seek.iter().position(|&c| c == b'\n').unwrap() + 1;
            let stop = (claim + REGION_CLAIM + 1).min(len);
            while line < stop {
                let word = read(input.add(line)) ^ 0x2C2C_2C2C_2C2C_2C2C;
                let comma =
                    word.wrapping_sub(0x0101_0101_0101_0101) & !word & 0x8080_8080_8080_8080;
                let digits = comma.trailing_zeros() as usize / 8;
                let comma = input.add(line + digits);
                let id = number_before(comma, digits) as usize;
                let code = [*comma.add(1), *comma.add(2), *comma.add(3)];
                let at = |c: u8| (c - b'A') as usize;
                let cell = &self.letters[at(code[0]) * 676 + at(code[1]) * 26 + at(code[2])];
                let group = match cell.load(Relaxed) {
                    0 => self.group_of_country(code, cell),
                    group => group,
                };
                assert!(id < REGION_IDS);
                *self.region_table.add(id) = group;
                line += digits + 5;
            }
        }
    }

    /// The first record at or after `claim`.
    unsafe fn first_record(&self, claim: usize) -> usize {
        let input = self.records;
        let mut at = claim.saturating_sub(1);
        loop {
            let rest = std::slice::from_raw_parts(input.add(at), self.records_len - at);
            let Some(found) = rest.iter().position(|&c| c == b'\n') else {
                return self.records_len;
            };
            at += found + 1;
            if at >= self.records_len || claim == 0 || (*input.add(at)).is_ascii_digit() {
                return at;
            }
            if *input.add(at) == 0xEF {
                return at;
            }
        }
    }

    unsafe fn parse_records(&self) {
        let input = self.records;
        let len = self.records_len;
        let mapped_end = len.div_ceil(4096) * 4096;
        let mut stats = vec![Stats::new(); GROUPS];
        let mut positions = [0u32; MAX_POSITIONS];
        let mut quotes = [0u64; QUOTE_WORDS];
        let mut quotes_before = [0u32; QUOTE_WORDS];
        let mut pending: Vec<Pending> = Vec::with_capacity(MAX_RECORDS);
        loop {
            let claim = self.next_claim.fetch_add(CLAIM, Relaxed);
            if claim >= len {
                break;
            }
            let claim_end = claim + CLAIM;
            madvise(
                input.add(claim) as *mut u8,
                CLAIM.min(mapped_end - claim),
                MADV_POPULATE_READ,
            );
            let origin = self.first_record(claim);
            let mut record = origin; // the next record to read
            let mut at = origin; // the next byte to scan
            let (mut inside, mut quote_total, mut word, mut filled) = (0u64, 0u32, 0usize, 0usize);
            while record < claim_end && record < len {
                for _ in 0..BATCH_WORDS {
                    if at >= len {
                        break;
                    }
                    let low = load(input.add(at));
                    let high = load(input.add(at + 32));
                    let mask =
                        |byte| equal_mask(low, byte) as u64 | (equal_mask(high, byte) as u64) << 32;
                    let (quote, comma, newline) = (mask(b'"'), mask(b','), mask(b'\n'));
                    let inside_now = prefix_xor(quote) ^ inside;
                    inside = ((inside_now as i64) >> 63) as u64;
                    let mut structural = (comma | newline) & !inside_now;
                    quotes[word % QUOTE_WORDS] = quote;
                    quotes_before[word % QUOTE_WORDS] = quote_total;
                    quote_total += quote.count_ones();
                    // Unconditional stores, so that a varying count needs no loop exit to
                    // predict. Eight cover almost every step.
                    let found = structural.count_ones() as usize;
                    for extra in 0..8 {
                        *positions.get_unchecked_mut(filled + extra) =
                            (at + structural.trailing_zeros() as usize) as u32;
                        structural &= structural.wrapping_sub(1);
                    }
                    let mut extra = 8;
                    while structural != 0 {
                        positions[filled + extra] =
                            (at + structural.trailing_zeros() as usize) as u32;
                        structural &= structural - 1;
                        extra += 1;
                    }
                    filled += found;
                    at += 64;
                    word += 1;
                }

                let quotes_up_to = |position: usize| {
                    let (word, bit) = ((position - origin) / 64, (position - origin) % 64);
                    let word = word % QUOTE_WORDS;
                    let below = _bzhi_u64(*quotes.get_unchecked(word), bit as u32);
                    quotes_before
                        .get_unchecked(word)
                        .wrapping_add(below.count_ones())
                };
                let complete = filled / FIELDS;
                pending.clear();
                // Nothing before a name's comma holds a quote except the record before, so
                // each record starts from where the last one's note ended.
                let mut rank = quotes_up_to(record);
                let mut read_positions = 0;
                while read_positions < complete * FIELDS && record < claim_end {
                    let field = positions.as_ptr().add(read_positions);
                    read_positions += FIELDS;
                    let start = input.add(record);
                    let comma1 = *field as usize;
                    let comma2 = *field.add(1) as usize;
                    let comma3 = *field.add(2) as usize;
                    let comma4 = *field.add(3) as usize;
                    let newline = *field.add(4) as usize;
                    record = newline + 1;

                    let bom = (*start == 0xEF) as usize * 3;
                    let id_end = input.add(comma1);
                    let id =
                        number_before(id_end, comma1 - (start as usize - input as usize) - bom);
                    let region = number_before(input.add(comma4), comma4 - comma3 - 1) as usize;
                    let region = region.min(REGION_IDS);
                    _mm_prefetch::<_MM_HINT_T0>(self.region_table.add(region).cast());
                    let end = newline - (*input.add(newline - 1) == b'\r') as usize;
                    let amount = cents_before(input.add(end), end - comma4 - 1);

                    let (rank2, rank3) = (quotes_up_to(comma2), quotes_up_to(comma3));
                    // A field of raw length L with q quote bytes has L - q/2 - 1 value bytes,
                    // or L when there are no quotes: q counts the delimiters and each pair.
                    let value_len = |raw: usize, q: u32| raw - (q / 2) as usize - (q != 0) as usize;
                    let raw_name = comma2 - comma1 - 1;
                    let name_quotes = rank2.wrapping_sub(rank);
                    let note_quotes = rank3.wrapping_sub(rank2);
                    rank = rank3;

                    let hash = name_hash(input.add(comma1 + 2), raw_name.saturating_sub(2));
                    let bucket = (hash >> (64 - BUCKETS.trailing_zeros())) as usize;
                    _mm_prefetch::<_MM_HINT_T0>(self.buckets.add(bucket).cast());
                    pending.push(Pending {
                        hash,
                        id,
                        amount,
                        region: region as u32,
                        name_bytes: value_len(raw_name, name_quotes) as u32,
                        note_bytes: value_len(comma3 - comma2 - 1, note_quotes) as u32,
                    });
                }
                // The region table and the name table were each a cache miss on every record,
                // half of the pass above when it read them itself. Now both are prefetched
                // there and read here. A locked update of a cold line stalls the core, so
                // its lines are fetched here and updated in the pass after.
                for name in &mut pending {
                    let group = *self.region_table.add(name.region as usize) as usize;
                    let s = stats.get_unchecked_mut(group);
                    s.count += 1;
                    s.id_sum += name.id;
                    s.amount_sum += name.amount;
                    s.amount_min = s.amount_min.min(name.amount);
                    s.amount_max = s.amount_max.max(name.amount);
                    s.name_bytes += name.name_bytes as u64;
                    s.note_bytes += name.note_bytes as u64;
                    let id = self.intern(name) as usize;
                    let word = group * SEEN_WORDS + id / 64;
                    _mm_prefetch::<_MM_HINT_ET0>(self.seen.add(word).cast());
                    name.hash = (word << 6 | id % 64) as u64;
                }
                for name in &pending {
                    (*self.seen.add((name.hash >> 6) as usize))
                        .fetch_or(1 << (name.hash & 63), Relaxed);
                }

                let left = filled - read_positions;
                positions.copy_within(read_positions..filled, 0);
                filled = left;
            }
        }
        let mut totals = self.totals.lock().unwrap();
        for (total, part) in totals.iter_mut().zip(&stats) {
            total.merge(part);
        }
    }

    /// The number of a name, new if this is its first appearance.
    #[inline(always)]
    unsafe fn intern(&self, name: &Pending) -> u32 {
        let index = (name.hash >> (64 - BUCKETS.trailing_zeros())) as usize;
        let slots = self.buckets.add(index).cast::<__m256i>();
        let wanted = _mm256_set1_epi64x(name.hash as i64);
        let low = _mm256_cmpeq_epi64(_mm256_load_si256(slots), wanted);
        let high = _mm256_cmpeq_epi64(_mm256_load_si256(slots.add(1)), wanted);
        let found = _mm256_movemask_pd(_mm256_castsi256_pd(low))
            | (_mm256_movemask_pd(_mm256_castsi256_pd(high)) & 1) << 4;
        if found != 0 {
            let number =
                (*self.buckets.add(index)).numbers[found.trailing_zeros() as usize].load(Relaxed);
            if number != 0 {
                return number - 1;
            }
        }
        self.insert(index, name.hash)
    }

    /// Finds or adds a name that its first bucket does not answer for. Only about 100,000
    /// lookups add one, and a compare-and-swap decides each slot, so no lock is needed.
    #[cold]
    unsafe fn insert(&self, mut index: usize, hash: u64) -> u32 {
        loop {
            let bucket = &*self.buckets.add(index);
            for (tag, cell) in bucket.tags.iter().zip(&bucket.numbers) {
                let current = tag.load(Relaxed);
                let won = current == 0 && tag.compare_exchange(0, hash, Relaxed, Relaxed).is_ok();
                if won {
                    let number = self.next_name.fetch_add(1, Relaxed) as u32 + 1;
                    assert!((number as usize) <= MAX_NAMES);
                    cell.store(number, Relaxed);
                    return number - 1;
                }
                if tag.load(Relaxed) == hash {
                    // Another worker may still be writing the number.
                    loop {
                        let number = cell.load(Relaxed);
                        if number != 0 {
                            return number - 1;
                        }
                    }
                }
            }
            index = (index + 1) % BUCKETS;
        }
    }

    fn work(&self) {
        unsafe { self.parse_regions() };
        self.regions_done.wait();
        unsafe { self.parse_records() };
    }
}

/// Cents as text: a sign if negative, the units, a point and exactly two digits.
fn money(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let (units, rest) = (cents.unsigned_abs() / 100, cents.unsigned_abs() % 100);
    format!("{sign}{units}.{rest:02}")
}

/// Maps a file for reading, with a spare zero page after it so wide loads may run past the end.
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

/// A table of whole huge pages, which must start on a huge page boundary to get any. It is
/// touched at once, because huge pages are about to be disabled for everything else.
unsafe fn zeroed_huge<T>(count: usize) -> *mut T {
    let bytes = count * size_of::<T>();
    let region: *mut u8 = zeroed(bytes + HUGE_PAGE);
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, bytes, MADV_HUGEPAGE);
    for page in (0..bytes).step_by(HUGE_PAGE) {
        *aligned.add(page) = 0;
    }
    aligned.cast()
}

unsafe fn zeroed<T>(count: usize) -> *mut T {
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    let bytes = count * size_of::<T>();
    mmap(
        std::ptr::null_mut(),
        bytes,
        PROT_READ | PROT_WRITE,
        flags,
        -1,
        0,
    )
    .cast()
}

fn main() {
    let records = File::open("/data/records.csv").unwrap();
    let regions = File::open("/data/regions.csv").unwrap();
    let records_len = records.metadata().unwrap().len() as usize;
    let regions_len = regions.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());

    let run = unsafe {
        let buckets = zeroed_huge(BUCKETS);
        let seen = zeroed_huge(GROUPS * SEEN_WORDS);
        prctl(PR_SET_THP_DISABLE, 1usize, 0usize, 0usize, 0usize);
        Run {
            records: map_file(&records, records_len),
            records_len,
            regions: map_file(&regions, regions_len),
            regions_len,
            next_region_claim: AtomicUsize::new(0),
            next_claim: AtomicUsize::new(0),
            region_table: zeroed(REGION_IDS + 1),
            letters: (0..LETTERS).map(|_| AtomicU8::new(0)).collect(),
            countries: Mutex::new(Vec::new()),
            regions_done: Barrier::new(threads),
            buckets,
            next_name: AtomicUsize::new(0),
            seen,
            totals: Mutex::new(vec![Stats::new(); GROUPS]),
        }
    };
    thread::scope(|s| {
        for _ in 1..threads {
            let builder = thread::Builder::new().stack_size(1 << 19);
            builder.spawn_scoped(s, || run.work()).unwrap();
        }
        run.work();
    });

    let mut totals = run.totals.lock().unwrap();
    for (group, total) in totals.iter_mut().enumerate() {
        let row =
            unsafe { std::slice::from_raw_parts(run.seen.add(group * SEEN_WORDS), SEEN_WORDS) };
        total.names_distinct = row
            .iter()
            .map(|word| word.load(Relaxed).count_ones() as u64)
            .sum();
    }
    let totals = totals;
    let mut lines: Vec<(String, &Stats)> = Vec::new();
    for (group, code) in run.countries.lock().unwrap().iter().enumerate() {
        lines.push((
            String::from_utf8_lossy(code).into_owned(),
            &totals[group + 1],
        ));
    }
    lines.push(("_UNMATCHED".to_string(), &totals[0]));
    lines.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::new();
    for (key, s) in lines {
        if s.count > 0 {
            out += &format!(
                "{key} {} {} {} {} {} {} {} {}\n",
                s.count,
                s.id_sum,
                money(s.amount_sum),
                money(s.amount_min),
                money(s.amount_max),
                s.names_distinct,
                s.name_bytes,
                s.note_bytes
            );
        }
    }
    File::create("/out/aggregates.txt")
        .unwrap()
        .write_all(out.as_bytes())
        .unwrap();
}

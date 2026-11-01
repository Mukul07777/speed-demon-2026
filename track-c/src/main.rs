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
use std::os::fd::AsRawFd;
use std::ptr::read_unaligned;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering::*};
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
// MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
const MAP_ANON: i32 = 0x4022;
const MADV_HUGEPAGE: i32 = 14;
const MADV_DONTNEED: i32 = 4;
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
// Digit arrays are decoded four at a time, so they may be read up to three past the last record.
const BATCH_CAP: usize = MAX_RECORDS + 3;
// The two numbers of a record that wait for `decode_digits`, as indexes of a batch's arrays.
const ID: usize = 0;
const AMOUNT: usize = 1;
const QUOTE_WORDS: usize = 1 << 10; // masks kept for the rank query, more than one batch
const BUCKETS: usize = 1 << 15;
const BUCKET_SLOTS: usize = 8;
const NUMBER_BITS: i32 = 17; // MAX_NAMES
const MAX_NAMES: usize = 1 << 17; // the generator draws from 100,000
const REGION_IDS: usize = 1_000_000; // lookup ids stay below this; anything above is unmatched
const GROUPS: usize = 256; // 240 countries, with index 0 for unmatched
const SEEN_WORDS: usize = MAX_NAMES / 64; // of one group's row of the name bit matrix
const LETTERS: usize = 26 * 26 * 26;
const ZEROS: u64 = 0x3030_3030_3030_3030;

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Stats {
    count: u64,
    id_sum: u64,
    amount_sum: i64,
    amount_min: i64,
    amount_max: i64,
    // Name bytes in the low half and note bytes in the high half: the spec keeps each below
    // 2^31, so the halves never carry into each other and one add updates both.
    lengths: u64,
}

impl Stats {
    fn new() -> Stats {
        let (amount_min, amount_max) = (i64::MAX, i64::MIN);
        Stats {
            amount_min,
            amount_max,
            ..Stats::default()
        }
    }

    fn merge(&mut self, other: &Stats) {
        self.count += other.count;
        self.id_sum += other.id_sum;
        self.amount_sum += other.amount_sum;
        self.amount_min = self.amount_min.min(other.amount_min);
        self.amount_max = self.amount_max.max(other.amount_max);
        self.lengths += other.lengths;
    }
}

/// One record between the passes of a batch. Its id and amount wait in the batch's digit
/// arrays until `decode_digits` has turned them into numbers.
#[derive(Clone, Copy)]
struct Pending {
    hash: u64,
    lengths: u64, // as in `Stats`
    sign: i64,    // -1 for a negative amount, else 0
    region: u32,
}

/// One cache line of eight slots. A slot holds a 47-bit name tag above the name's dense number,
/// so one compare-and-swap claims a slot with its number. Numbers keep the bit matrix small.
#[repr(C, align(64))]
struct Bucket {
    slots: [AtomicU64; BUCKET_SLOTS],
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
    // Start addresses of the huge pages of the two tables, touched by whichever worker claims them.
    huge_pages: Vec<usize>,
    next_huge_page: AtomicUsize,
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
    eight_digits(digits_before(end, len))
}

/// The `len` digit bytes before `end`, for lengths 1 to 8, as `decode_digits` wants them.
/// Shifting them to the top makes the bytes below read as leading zeros.
unsafe fn digits_before(end: *const u8, len: usize) -> u64 {
    (read(end.sub(8)) ^ ZEROS) >> (64 - 8 * len) << (64 - 8 * len)
}

/// The digits of an amount such as `5.25` in the `len` bytes before `end`, without its point.
unsafe fn cents_digits(end: *const u8, len: usize) -> u64 {
    let bytes = digits_before(end, len);
    // Close the gap the point leaves: its byte is always third from the end.
    (bytes & 0xFF_FFFF_FFFF) << 8 | bytes & 0xFFFF_0000_0000_0000
}

/// Turns the first `count` words of eight digit bytes each into numbers, four at a time, like
/// `eight_digits` does for one. The three arrays of a batch go through the same few
/// instructions, so a record's numbers cost far less than three scalar multiply chains.
unsafe fn decode_digits(words: &[u64; BATCH_CAP], numbers: &mut [u32; BATCH_CAP], count: usize) {
    for at in (0..count).step_by(4) {
        let digits = _mm256_and_si256(load(words.as_ptr().add(at).cast()), _mm256_set1_epi8(15));
        // Pairs of digits, then groups of four, then all eight.
        let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
        let fours = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x0001_0064));
        let eights = _mm256_madd_epi16(
            _mm256_packus_epi32(fours, fours),
            _mm256_set1_epi32(0x0001_2710),
        );
        // Each 128-bit half now holds its two numbers in its low 64 bits.
        let packed = _mm256_permute4x64_epi64::<0b1000>(eights);
        _mm_storeu_si128(
            numbers.as_mut_ptr().add(at).cast(),
            _mm256_castsi256_si128(packed),
        );
    }
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

/// The hash of a name, never 0, which marks an unused slot, and the number of quote bytes in
/// it. Names are compared by their raw bytes: every non-empty name is quoted, so equal raw
/// bytes mean equal values and nothing needs unescaping. Names of 32 bytes and more do not
/// occur at seed 1 (the longest is 27), so they cost one predicted branch here.
unsafe fn name_hash(name: *const u8, len: usize) -> (u64, usize) {
    // 32 bytes of ones then 32 of zeros: loading at 32 - len keeps the first `len` bytes.
    static MASKS: [[u8; 32]; 2] = [[0xFF; 32], [0; 32]];
    if len >= 32 {
        return long_name_hash(name, len);
    }
    let bytes = _mm256_and_si256(load(name), load(MASKS.as_ptr().cast::<u8>().add(32 - len)));
    let quotes = equal_mask(bytes, b'"').count_ones() as usize;
    let key = _mm_set_epi64x(0x1D8E_4E27_C47D_124F, len as i64 ^ 0x5899_65CC_7537_4CC3);
    let mixed = _mm_aesenc_si128(
        _mm_xor_si128(_mm256_castsi256_si128(bytes), key),
        _mm256_extracti128_si256::<1>(bytes),
    );
    let mixed = _mm_aesenc_si128(mixed, _mm_set1_epi64x(0xA076_1D64_78BD_642F_u64 as i64));
    let mixed = _mm_aesenc_si128(mixed, _mm_set1_epi64x(0xEB44_ACCA_B455_D165_u64 as i64));
    (_mm_cvtsi128_si64(mixed) as u64 | 1, quotes)
}

#[cold]
unsafe fn long_name_hash(name: *const u8, len: usize) -> (u64, usize) {
    let bytes = std::slice::from_raw_parts(name, len);
    let hash = bytes.iter().fold(len as u64, |hash, &byte| {
        (hash.rotate_left(23) ^ byte as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)
    });
    (hash | 1, bytes.iter().filter(|&&byte| byte == b'"').count())
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
                let letters = code.iter().fold(0, |n, &c| n * 26 + (c - b'A') as usize);
                let cell = &self.letters[letters];
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
        let rest = std::slice::from_raw_parts(self.records, self.records_len);
        let mut at = claim.saturating_sub(1);
        while let Some(found) = rest[at..].iter().position(|&c| c == b'\n') {
            at += found + 1;
            if at >= rest.len() || claim == 0 || matches!(rest[at], b'0'..=b'9' | 0xEF) {
                return at;
            }
        }
        rest.len()
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
        let mut digits = [[0u64; BATCH_CAP]; 2];
        let mut numbers = [[0u32; BATCH_CAP]; 2];
        loop {
            let claim = self.next_claim.fetch_add(CLAIM, Relaxed);
            if claim >= len {
                break;
            }
            let claim_end = claim + CLAIM;
            let advise = |advice| {
                madvise(
                    input.add(claim) as *mut u8,
                    CLAIM.min(mapped_end - claim),
                    advice,
                )
            };
            advise(MADV_POPULATE_READ);
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
                    let field = positions
                        .as_ptr()
                        .add(read_positions)
                        .cast::<[u32; FIELDS]>();
                    read_positions += FIELDS;
                    let [comma1, comma2, comma3, comma4, newline] = (*field).map(|p| p as usize);
                    let bom = (*input.add(record) == 0xEF) as usize * 3;
                    let id_len = comma1 - record - bom;
                    record = newline + 1;

                    let at = pending.len();
                    digits[ID][at] = digits_before(input.add(comma1), id_len);
                    let region = number_before(input.add(comma4), comma4 - comma3 - 1) as usize;
                    let region = region.min(REGION_IDS);
                    _mm_prefetch::<_MM_HINT_T0>(self.region_table.add(region).cast());
                    // The sign sits before the digits, where the shift leaves it out.
                    let end = newline - (*input.add(newline - 1) == b'\r') as usize;
                    let negative = (*input.add(comma4 + 1) == b'-') as usize;
                    digits[AMOUNT][at] = cents_digits(input.add(end), end - comma4 - 1 - negative);

                    // The name has just been read, so only the note's end needs a rank query.
                    let raw_name = comma2 - comma1 - 1;
                    let (hash, name_inside) =
                        name_hash(input.add(comma1 + 2), raw_name.saturating_sub(2));
                    let name_len = raw_name.saturating_sub(2) - name_inside / 2;
                    let rank3 = quotes_up_to(comma3);
                    // The delimiters add two quotes, but only to a name that is not empty.
                    let name_quotes = name_inside + 2 * (raw_name != 0) as usize;
                    let note_quotes = rank3.wrapping_sub(rank) as usize - name_quotes;
                    rank = rank3;
                    // A note of raw length L with q quote bytes has L - q/2 - 1 value bytes, or
                    // L when there are no quotes: q counts the delimiters and each pair.
                    let note_len =
                        comma3 - comma2 - 1 - note_quotes / 2 - (note_quotes != 0) as usize;
                    let bucket = (hash >> (64 - BUCKETS.trailing_zeros())) as usize;
                    _mm_prefetch::<_MM_HINT_T0>(self.buckets.add(bucket).cast());
                    pending.push(Pending {
                        hash,
                        sign: -(negative as i64),
                        region: region as u32,
                        lengths: name_len as u64 | (note_len as u64) << 32,
                    });
                }
                // The region table and the name table were each a cache miss on every record,
                // half of the pass above when it read them itself. Now both are prefetched
                // before this pass reads them. A locked update of a cold line stalls the core,
                // so its lines are fetched here and updated in the pass after.
                for kind in [ID, AMOUNT] {
                    decode_digits(&digits[kind], &mut numbers[kind], pending.len());
                }
                for (at, name) in pending.iter_mut().enumerate() {
                    let group = *self.region_table.add(name.region as usize) as usize;
                    let amount = (numbers[AMOUNT][at] as i64 ^ name.sign) - name.sign;
                    let s = stats.get_unchecked_mut(group);
                    s.count += 1;
                    s.id_sum += numbers[ID][at] as u64;
                    s.amount_sum += amount;
                    s.amount_min = s.amount_min.min(amount);
                    s.amount_max = s.amount_max.max(amount);
                    s.lengths += name.lengths;
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
            // Unmap this claim's pages now, in parallel. Otherwise the kernel frees all
            // ~50,000 page-table entries on one thread at exit, which the timer includes. A
            // record running into the next claim just faults its few pages back in.
            advise(MADV_DONTNEED);
        }
        let mut totals = self.totals.lock().unwrap();
        for (total, part) in totals.iter_mut().zip(&stats) {
            total.merge(part);
        }
    }

    /// The number of a name, new if this is its first appearance.
    unsafe fn intern(&self, name: &Pending) -> u32 {
        let index = (name.hash >> (64 - BUCKETS.trailing_zeros())) as usize;
        let tag = name.hash & (1 << 47) - 1 | 1;
        let slots = self.buckets.add(index).cast::<__m256i>();
        let wanted = _mm256_set1_epi64x(tag as i64);
        let [low, high] = [0, 1].map(|half| {
            let slot = _mm256_srli_epi64::<NUMBER_BITS>(_mm256_load_si256(slots.add(half)));
            _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(slot, wanted)))
        });
        let found = low | high << 4;
        if found != 0 {
            let slot =
                (*self.buckets.add(index)).slots[found.trailing_zeros() as usize].load(Relaxed);
            return slot as u32 & (MAX_NAMES - 1) as u32;
        }
        self.insert(index, tag)
    }

    /// Adds a name whose bucket does not hold it: the 100,000 first appearances, and the few
    /// whose bucket is full and spills into the next. A compare-and-swap decides each slot, so
    /// no lock is needed. A worker that loses a race to the same name wastes one number.
    #[cold]
    unsafe fn insert(&self, mut index: usize, tag: u64) -> u32 {
        loop {
            for slot in &(*self.buckets.add(index)).slots {
                let mut seen = slot.load(Relaxed);
                if seen == 0 {
                    let number = self.next_name.fetch_add(1, Relaxed);
                    assert!(number < MAX_NAMES);
                    seen = match slot.compare_exchange(
                        0,
                        tag << NUMBER_BITS | number as u64,
                        Relaxed,
                        Relaxed,
                    ) {
                        Ok(_) => return number as u32,
                        Err(other) => other,
                    };
                }
                if seen >> NUMBER_BITS == tag {
                    return seen as u32 & (MAX_NAMES - 1) as u32;
                }
            }
            index = (index + 1) % BUCKETS;
        }
    }

    /// Starts `descendants` more workers as a tree, so that no thread spends a millisecond
    /// spawning them all, and then works itself.
    fn start<'scope>(
        &'scope self,
        scope: &'scope thread::Scope<'scope, '_>,
        mut descendants: usize,
    ) {
        while descendants > 0 {
            let share = descendants.div_ceil(2);
            scope.spawn(move || self.start(scope, share - 1));
            descendants -= share;
        }
        self.work();
    }

    fn work(&self) {
        // Zeroing a huge page takes a few hundred microseconds, so the workers share it and do
        // it while the regions load. It must end before huge pages are disabled.
        let page = self.next_huge_page.fetch_add(1, Relaxed);
        if let Some(&at) = self.huge_pages.get(page) {
            unsafe { *(at as *mut u8) = 0 };
        }
        unsafe { self.parse_regions() };
        self.regions_done.wait();
        // Idempotent, and set before this worker allocates anything for the records.
        unsafe { prctl(PR_SET_THP_DISABLE, 1usize, 0usize, 0usize, 0usize) };
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
    let base = mmap(std::ptr::null_mut(), span, PROT_READ, MAP_ANON, -1, 0);
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

/// A table of whole huge pages, which must start on a huge page boundary to get any. Its pages
/// are touched later by the workers, because huge pages are disabled for everything else
/// before the records start. Returns the table and the start of each of its pages.
unsafe fn zeroed_huge<T>(count: usize) -> (*mut T, impl Iterator<Item = usize>) {
    let bytes = count * size_of::<T>();
    let region: *mut u8 = zeroed(bytes + HUGE_PAGE);
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, bytes, MADV_HUGEPAGE);
    let start = aligned as usize;
    (aligned.cast(), (start..start + bytes).step_by(HUGE_PAGE))
}

unsafe fn zeroed<T>(count: usize) -> *mut T {
    let bytes = count * size_of::<T>();
    mmap(
        std::ptr::null_mut(),
        bytes,
        PROT_READ | PROT_WRITE,
        MAP_ANON,
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
        let (buckets, bucket_pages) = zeroed_huge(BUCKETS);
        let (seen, seen_pages) = zeroed_huge(GROUPS * SEEN_WORDS);
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
            huge_pages: bucket_pages.chain(seen_pages).collect(),
            next_huge_page: AtomicUsize::new(0),
            buckets,
            next_name: AtomicUsize::new(0),
            seen,
            totals: Mutex::new(vec![Stats::new(); GROUPS]),
        }
    };
    thread::scope(|s| {
        run.start(s, threads - 1);
    });

    let totals = run.totals.lock().unwrap();
    let codes = run.countries.lock().unwrap();
    let mut keys: Vec<_> = (codes.iter().zip(1..))
        .map(|(code, group)| (std::str::from_utf8(code).unwrap(), group))
        .collect();
    keys.sort_unstable();
    // `_UNMATCHED` sorts after every country code.
    keys.push(("_UNMATCHED", 0));
    let mut out = String::new();
    for (key, group) in keys {
        let s = &totals[group];
        let row =
            unsafe { std::slice::from_raw_parts(run.seen.add(group * SEEN_WORDS), SEEN_WORDS) };
        let distinct = row
            .iter()
            .map(|w| w.load(Relaxed).count_ones())
            .sum::<u32>();
        if s.count > 0 {
            out += &format!(
                "{key} {} {} {} {} {} {} {} {}\n",
                s.count,
                s.id_sum,
                money(s.amount_sum),
                money(s.amount_min),
                money(s.amount_max),
                distinct,
                s.lengths & 0xFFFF_FFFF,
                s.lengths >> 32
            );
        }
    }
    std::fs::write("/out/aggregates.txt", out).unwrap();
}

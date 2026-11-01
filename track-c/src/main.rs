//! Speed Demon 2026, Track C (Join).
//!
//! The file is parsed in parallel without ever scanning from its start, because the spec
//! gives a safe resync point: inside a note an LF is only ever followed by a lowercase word
//! or a quote, so an LF followed by a digit or a BOM always starts a record.
//!
//! Each worker claims 1 MiB and walks it in 2 KiB batches:
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
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::ptr::{null_mut, read_unaligned};
use std::slice::from_raw_parts;
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
// MAP_PRIVATE | MAP_FIXED
const MAP_FILE: i32 = 0x12;
// MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
const MAP_ANON: i32 = 0x4022;
const MADV_HUGEPAGE: i32 = 14;
const MADV_DONTNEED: i32 = 4;
const PR_SET_THP_DISABLE: i32 = 41;

const HUGE_PAGE: usize = 1 << 21;
const CLAIM: usize = 1 << 20;
const REGION_CLAIM: usize = 1 << 16;
const BATCH_WORDS: usize = 32; // 64-byte steps of the file between record passes
const FIELDS: usize = 5; // structural bytes per record: four commas and an LF
                         // Records that fit in a batch (the shortest is `1,,,1,0.00` and an LF, 11 bytes), two more that
                         // were left over from the one before, and three for the digit arrays, which are decoded four at
                         // a time and so may be read up to three past the last record.
const MAX_RECORDS: usize = BATCH_WORDS * 64 / 11 + 5;
// The three numbers of a record that wait for `decode_digits`, as indexes of a batch's arrays.
const ID: usize = 0;
const AMOUNT: usize = 1;
const REGION: usize = 2;
const QUOTE_WORDS: usize = 1 << 10; // masks kept for the rank query, more than one batch
const BUCKETS: usize = 1 << 15;
const NAME_BLOCK: usize = 32; // name numbers a worker reserves at a time
const NUMBER_BITS: i32 = 17; // a name's dense number; the generator draws from 100,000 names
const NUMBER_MASK: u64 = (1 << NUMBER_BITS) - 1;
const REGION_IDS: usize = 1_000_000; // lookup ids stay below this; anything above is unmatched
const GROUPS: usize = 256; // 240 countries, with index 0 for unmatched
const SEEN_WORDS: usize = 1 << (NUMBER_BITS - 6); // of one group's row of the name bit matrix
                                                  // Amounts are kept as the largest of `BIAS + amount` and of `BIAS - amount`, both positive, so
                                                  // that a zeroed `Stats` is a valid empty one and merging two needs only sums and maxima.
const BIAS: i64 = 1 << 40;

static NEXT_REGION_CLAIM: AtomicUsize = AtomicUsize::new(0);
static NEXT_CLAIM: AtomicUsize = AtomicUsize::new(0);
static NEXT_HUGE_PAGE: AtomicUsize = AtomicUsize::new(0);
static NEXT_NAME: AtomicUsize = AtomicUsize::new(0);
// Region id to group, and country code (as base-26 digits) to group. Both start zeroed.
static REGION_GROUP: [AtomicU8; REGION_IDS + 1] = [const { AtomicU8::new(0) }; REGION_IDS + 1];
static LETTERS: [AtomicU8; 26 * 26 * 26] = [const { AtomicU8::new(0) }; 26 * 26 * 26];
// The sums of all workers, in groups.
static TOTALS: Mutex<[Stats; GROUPS]> = Mutex::new(unsafe { std::mem::zeroed() });
// Group to country code, with the unmatched group first.
static COUNTRIES: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Stats {
    count: u64,
    id_sum: u64,
    amount_sum: i64,
    amount_hi: i64, // the largest `BIAS + amount`
    amount_lo: i64, // the largest `BIAS - amount`
    // Name bytes in the low half and note bytes in the high half: the spec keeps each below
    // 2^31, so the halves never carry into each other and one add updates both.
    lengths: u64,
}

/// One record between the passes of a batch. Its id and amount wait in the batch's digit
/// arrays until `decode_digits` has turned them into numbers.
#[derive(Clone, Copy, Default)]
struct Pending {
    hash: u64,
    lengths: u64, // as in `Stats`
    sign: i64,    // -1 for a negative amount, else 0
    region: u32,
}

/// One cache line of eight slots. A slot holds a 47-bit name tag above the name's dense number,
/// so one compare-and-swap claims a slot with its number. Numbers keep the bit matrix small.
type Bucket = [AtomicU64; 8];

/// The two shared tables, zeroed. Together they are exactly three huge pages.
#[repr(C)]
struct Tables {
    buckets: [Bucket; BUCKETS],
    seen: [AtomicU64; GROUPS * SEEN_WORDS], // the name bit matrix, SEEN_WORDS per group
}

struct Run {
    records: &'static [u8],
    regions: &'static [u8],
    tables: &'static Tables,
    regions_done: Barrier,
}

unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p.cast())
}

unsafe fn equal_mask(chunk: __m256i, byte: u8) -> u64 {
    _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(byte as i8))) as u32 as u64
}

/// The `len` digit bytes before `end`, for lengths 1 to 8, as `decode_digits` wants them.
/// Shifting them to the top makes the bytes below read as leading zeros.
unsafe fn digits_before(end: *const u8, len: usize) -> u64 {
    (read_unaligned(end.sub(8).cast::<u64>()) ^ 0x3030_3030_3030_3030) >> (64 - 8 * len)
        << (64 - 8 * len)
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
unsafe fn decode_digits(
    words: &[u64; MAX_RECORDS],
    numbers: &mut [u32; MAX_RECORDS],
    count: usize,
) {
    let (ten, hundred, ten_thousand) = (0x010A, 0x0001_0064, 0x0001_2710);
    for at in (0..count).step_by(4) {
        let digits = _mm256_and_si256(load(words.as_ptr().add(at).cast()), _mm256_set1_epi8(15));
        // Pairs of digits, then groups of four, then all eight.
        let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(ten));
        let fours = _mm256_madd_epi16(pairs, _mm256_set1_epi32(hundred));
        let fours = _mm256_packus_epi32(fours, fours);
        let eights = _mm256_madd_epi16(fours, _mm256_set1_epi32(ten_thousand));
        // Each 128-bit half now holds its two numbers in its low 64 bits.
        let packed = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(eights));
        _mm_storeu_si128(numbers.as_mut_ptr().add(at).cast(), packed);
    }
}

/// Sets each bit that has an odd number of set bits at or below it, so quote bits become a
/// mask of everything inside quotes.
unsafe fn prefix_xor(bits: u64) -> u64 {
    let (bits, ones) = (_mm_cvtsi64_si128(bits as i64), _mm_set1_epi8(-1));
    _mm_cvtsi128_si64(_mm_clmulepi64_si128(bits, ones, 0)) as u64
}

/// The hash of a name, never 0, which marks an unused slot, and the number of quote bytes in
/// it. Names are compared by their raw bytes: every non-empty name is quoted, so equal raw
/// bytes mean equal values and nothing needs unescaping. Names of 32 bytes and more do not
/// occur at seed 1 (the longest is 27), so they cost one predicted branch here.
unsafe fn name_hash(name: *const u8, len: usize) -> (u64, usize) {
    // 32 bytes of ones then 32 of zeros: loading at 32 - len keeps the first `len` bytes.
    static MASKS: [[u8; 32]; 2] = [[0xFF; 32], [0; 32]];
    if len >= 32 {
        return long_name_hash(from_raw_parts(name, len));
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
fn long_name_hash(name: &[u8]) -> (u64, usize) {
    let hash = (name.iter()).fold(name.len() as u64, |hash, &byte| {
        (hash.rotate_left(23) ^ byte as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)
    });
    (hash | 1, name.iter().filter(|&&byte| byte == b'"').count())
}

/// The next unclaimed block of `size` bytes in a file of `len` bytes, if any is left.
fn take_block(counter: &AtomicUsize, size: usize, len: usize) -> Option<usize> {
    Some(counter.fetch_add(size, Relaxed)).filter(|&at| at < len)
}

/// The group of a country, numbered from 1 in order of first appearance.
fn group_of(code: &[u8]) -> u8 {
    let cell = &LETTERS[code.iter().fold(0, |n, &c| n * 26 + (c - b'A') as usize)];
    if cell.load(Relaxed) == 0 {
        // Only 240 calls get this far, so a lock is fine.
        let mut countries = COUNTRIES.lock().unwrap();
        if cell.load(Relaxed) == 0 {
            countries.push(String::from_utf8_lossy(code).into_owned());
            cell.store(countries.len() as u8 - 1, Relaxed);
        }
    }
    cell.load(Relaxed)
}

impl Run {
    fn parse_regions(&self) {
        let text = self.regions;
        while let Some(claim) = take_block(&NEXT_REGION_CLAIM, REGION_CLAIM, text.len()) {
            // Each claim starts after the first LF at or past its start, so the header goes
            // to nobody and a line belongs to the claim holding the LF before it.
            let rest = &text[claim..];
            let mut line = rest.iter().position(|&c| c == b'\n').unwrap() + 1;
            while line <= REGION_CLAIM && line < rest.len() {
                let fields = &rest[line..];
                let digits = fields.iter().position(|&c| c == b',').unwrap();
                let id = fields[..digits]
                    .iter()
                    .fold(0, |n, &c| n * 10 + (c - b'0') as usize);
                REGION_GROUP[id].store(group_of(&fields[digits + 1..][..3]), Relaxed);
                line += digits + 5;
            }
        }
    }

    /// The first record at or after `claim`.
    fn first_record(&self, claim: usize) -> usize {
        let from = claim.saturating_sub(1);
        let begins = |pair: &[u8]| pair[0] == b'\n' && matches!(pair[1], b'0'..=b'9' | 0xEF);
        let found = self.records[from..].windows(2).position(begins);
        found.map_or(self.records.len(), |at| from + at + 1)
    }

    unsafe fn parse_records(&self) {
        let input = self.records.as_ptr() as *mut u8;
        let len = self.records.len();
        let mut stats = [Stats::default(); GROUPS];
        let mut positions = [0u32; MAX_RECORDS * FIELDS];
        let mut quotes = [0u64; QUOTE_WORDS];
        let mut quotes_before = [0u32; QUOTE_WORDS];
        let mut pending = [Pending::default(); MAX_RECORDS];
        let mut spare = 0..0; // name numbers this worker reserved
        let mut digits = [[0u64; MAX_RECORDS]; 3];
        let mut numbers = [[0u32; MAX_RECORDS]; 3];
        while let Some(claim) = take_block(&NEXT_CLAIM, CLAIM, len) {
            let end = (claim + CLAIM).min(len);
            let origin = self.first_record(claim);
            let mut record = origin; // the next record to read
            let mut at = origin; // the next byte to scan
            let (mut inside, mut quote_total, mut filled) = (0u64, 0u32, 0usize);
            while record < end {
                for _ in 0..BATCH_WORDS {
                    let low = load(input.add(at));
                    let high = load(input.add(at + 32));
                    let mask = |byte| equal_mask(low, byte) | equal_mask(high, byte) << 32;
                    let (quote, comma, newline) = (mask(b'"'), mask(b','), mask(b'\n'));
                    let inside_now = prefix_xor(quote) ^ inside;
                    inside = ((inside_now as i64) >> 63) as u64;
                    let mut structural = (comma | newline) & !inside_now;
                    let word = (at - origin) / 64 % QUOTE_WORDS;
                    quotes[word] = quote;
                    quotes_before[word] = quote_total;
                    quote_total += quote.count_ones();
                    // Unconditional stores, so that a varying count needs no loop exit to
                    // predict. Eight cover almost every step.
                    let found = structural.count_ones() as usize;
                    let mut extra = 0;
                    while extra < 8 || structural != 0 {
                        *positions.get_unchecked_mut(filled + extra) =
                            (at + structural.trailing_zeros() as usize) as u32;
                        structural &= structural.wrapping_sub(1);
                        extra += 1;
                    }
                    filled += found;
                    at += 64;
                }

                let quotes_up_to = |position: usize| {
                    let word = (position - origin) / 64 % QUOTE_WORDS;
                    let below = _bzhi_u64(quotes[word], ((position - origin) % 64) as u32);
                    quotes_before[word].wrapping_add(below.count_ones())
                };
                let complete = filled / FIELDS;
                let field = |at: usize| {
                    let five = positions.as_ptr().add(at * FIELDS).cast::<[u32; FIELDS]>();
                    (*five).map(|p| p as usize)
                };
                // Pass A: how many records start inside this claim, and their numbers.
                let mut count = 0;
                let mut start = record;
                while count < complete && start < end {
                    let [comma1, _, comma3, comma4, newline] = field(count);
                    let bom = (*input.add(start) == 0xEF) as usize * 3;
                    digits[ID][count] = digits_before(input.add(comma1), comma1 - start - bom);
                    digits[REGION][count] = digits_before(input.add(comma4), comma4 - comma3 - 1);
                    let stop = newline - (*input.add(newline - 1) == b'\r') as usize;
                    let negative = (*input.add(comma4 + 1) == b'-') as usize;
                    digits[AMOUNT][count] =
                        cents_digits(input.add(stop), stop - comma4 - 1 - negative);
                    pending.get_unchecked_mut(count).sign = -(negative as i64);
                    start = newline + 1;
                    count += 1;
                }
                for kind in [ID, AMOUNT, REGION] {
                    decode_digits(&digits[kind], &mut numbers[kind], count);
                }
                // Pass B: names and notes.
                let mut rank = quotes_up_to(record);
                for at in 0..count {
                    let [comma1, comma2, comma3, ..] = field(at);
                    let raw_name = comma2 - comma1 - 1;
                    let (hash, name_inside) =
                        name_hash(input.add(comma1 + 2), raw_name.saturating_sub(2));
                    let name_len = raw_name.saturating_sub(2) - name_inside / 2;
                    let rank3 = quotes_up_to(comma3);
                    let name_quotes = name_inside + 2 * (raw_name != 0) as usize;
                    let note_quotes = rank3.wrapping_sub(rank) as usize - name_quotes;
                    rank = rank3;
                    let note_len =
                        comma3 - comma2 - 1 - note_quotes / 2 - (note_quotes != 0) as usize;
                    let bucket = (hash >> (64 - BUCKETS.trailing_zeros())) as usize;
                    _mm_prefetch::<_MM_HINT_T0>(self.tables.buckets.as_ptr().add(bucket).cast());
                    let region = (numbers[REGION][at] as usize).min(REGION_IDS);
                    _mm_prefetch::<_MM_HINT_T0>(REGION_GROUP.as_ptr().add(region).cast());
                    let slot = pending.get_unchecked_mut(at);
                    slot.region = region as u32;
                    slot.hash = hash;
                    slot.lengths = name_len as u64 | (note_len as u64) << 32;
                }
                let read_positions = count * FIELDS;
                record = start;
                // The region table and the name table were each a cache miss on every record,
                // half of the pass above when it read them itself. Now both are prefetched
                // before this pass reads them. A locked update of a cold line stalls the core,
                // so its lines are fetched here and updated in the pass after.
                for (at, name) in pending[..count].iter_mut().enumerate() {
                    let group = REGION_GROUP
                        .get_unchecked(name.region as usize)
                        .load(Relaxed) as usize;
                    let amount = (numbers[AMOUNT][at] as i64 ^ name.sign) - name.sign;
                    let s = stats.get_unchecked_mut(group);
                    s.count += 1;
                    s.id_sum += numbers[ID][at] as u64;
                    s.amount_sum += amount;
                    s.amount_hi = s.amount_hi.max(BIAS + amount);
                    s.amount_lo = s.amount_lo.max(BIAS - amount);
                    s.lengths += name.lengths;
                    let id = self.intern(name, &mut spare);
                    let word = group * SEEN_WORDS + id / 64;
                    _mm_prefetch::<_MM_HINT_ET0>(self.tables.seen.as_ptr().add(word).cast());
                    name.hash = (word << 6 | id % 64) as u64;
                }
                for name in &pending[..count] {
                    let word = self.tables.seen.get_unchecked((name.hash >> 6) as usize);
                    word.fetch_or(1 << (name.hash & 63), Relaxed);
                }

                let left = filled - read_positions;
                positions.copy_within(read_positions..filled, 0);
                filled = left;
            }
            // Unmap this claim's pages now, in parallel. Otherwise the kernel frees all
            // ~50,000 page-table entries on one thread at exit, which the timer includes. A
            // record running into the next claim just faults its few pages back in.
            // The length stops at the file's end (the kernel rounds it up to a page): madvise goes on
            // into whatever is mapped after the range, and would wipe it.
            madvise(input.add(claim), end - claim, MADV_DONTNEED);
        }
        let mut totals = TOTALS.lock().unwrap();
        for (total, part) in totals.iter_mut().zip(&stats) {
            total.count += part.count;
            total.id_sum += part.id_sum;
            total.amount_sum += part.amount_sum;
            total.amount_hi = total.amount_hi.max(part.amount_hi);
            total.amount_lo = total.amount_lo.max(part.amount_lo);
            total.lengths += part.lengths;
        }
    }

    /// The number of a name, new if this is its first appearance.
    unsafe fn intern(&self, name: &Pending, spare: &mut Range<usize>) -> usize {
        let index = (name.hash >> (64 - BUCKETS.trailing_zeros())) as usize;
        let tag = name.hash & (1 << 47) - 1 | 1;
        let bucket = &self.tables.buckets[index];
        let wanted = _mm256_set1_epi64x(tag as i64);
        let [low, high] = [0, 1].map(|half| {
            let slots = _mm256_load_si256(bucket.as_ptr().cast::<__m256i>().add(half));
            let slots = _mm256_srli_epi64::<NUMBER_BITS>(slots);
            _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(slots, wanted)))
        });
        let found = low | high << 4;
        if found == 0 {
            return self.insert(index, tag, spare);
        }
        (bucket[found.trailing_zeros() as usize].load(Relaxed) & NUMBER_MASK) as usize
    }

    /// Adds a name whose bucket does not hold it: the 100,000 first appearances, and the few
    /// whose bucket is full and spills into the next. A compare-and-swap decides each slot, so
    /// no lock is needed. A worker that loses a race to the same name wastes one number.
    #[cold]
    unsafe fn insert(&self, mut index: usize, tag: u64, spare: &mut Range<usize>) -> usize {
        loop {
            for slot in &self.tables.buckets[index] {
                if spare.start == spare.end {
                    let first = NEXT_NAME.fetch_add(NAME_BLOCK, Relaxed);
                    *spare = first..first + NAME_BLOCK;
                }
                // A plain read first: a CAS on a slot that is already taken is a wasted locked op.
                let held = slot.load(Relaxed);
                if held != 0 {
                    if held >> NUMBER_BITS == tag {
                        return (held & NUMBER_MASK) as usize;
                    }
                    continue;
                }
                let number = spare.start;
                let claimed = tag << NUMBER_BITS | number as u64;
                // A slot that is taken holds another name, or this one if a worker just added it.
                let seen = match slot.compare_exchange(0, claimed, Relaxed, Relaxed) {
                    Ok(_) => {
                        spare.start += 1;
                        return number;
                    }
                    Err(seen) => seen,
                };
                if seen >> NUMBER_BITS == tag {
                    return (seen & NUMBER_MASK) as usize;
                }
            }
            index = (index + 1) % BUCKETS;
        }
    }

    /// Starts `descendants` more workers as a tree, so that no thread spends a millisecond
    /// spawning them all, and then works itself.
    fn start<'a>(&'a self, scope: &'a thread::Scope<'a, '_>, mut descendants: usize) {
        while descendants > 0 {
            let share = descendants.div_ceil(2);
            scope.spawn(move || self.start(scope, share - 1));
            descendants -= share;
        }
        // Zeroing a huge page takes a few hundred microseconds, so the workers share it and do
        // it while the regions load. It must end before huge pages are disabled.
        let page = NEXT_HUGE_PAGE.fetch_add(1, Relaxed) * HUGE_PAGE;
        let bytes = self.tables as *const Tables as *const AtomicU8;
        if page < size_of::<Tables>() {
            unsafe { (*bytes.add(page)).store(0, Relaxed) };
        }
        self.parse_regions();
        self.regions_done.wait();
        // Idempotent, and set before this worker allocates anything for the records.
        unsafe { prctl(PR_SET_THP_DISABLE, 1usize, 0usize, 0usize, 0usize) };
        unsafe { self.parse_records() };
    }
}

/// Cents as text: a sign if negative, the units, a point and exactly two digits.
fn money(cents: i64) -> String {
    let (sign, abs) = (if cents < 0 { "-" } else { "" }, cents.unsigned_abs());
    format!("{sign}{}.{:02}", abs / 100, abs % 100)
}

/// An anonymous zeroed mapping.
unsafe fn anonymous(len: usize, prot: i32) -> *mut u8 {
    mmap(null_mut(), len, prot, MAP_ANON, -1, 0)
}

/// Maps a file for reading, with a spare zero page after it so wide loads may run past the end.
unsafe fn map_file(path: &str) -> &'static [u8] {
    let file = File::open(path).unwrap();
    let (fd, len) = (file.as_raw_fd(), file.metadata().unwrap().len() as usize);
    let base = anonymous(len.next_multiple_of(4096) + 4096, PROT_READ);
    mmap(base, len, PROT_READ, MAP_FILE, fd, 0);
    from_raw_parts(base, len)
}

/// A zeroed table of whole huge pages, which must start on a huge page boundary to get any. Its
/// pages are touched later by the workers, because huge pages are disabled for everything else
/// before the records start.
unsafe fn zeroed_huge<T>() -> &'static T {
    let region = anonymous(size_of::<T>() + HUGE_PAGE, PROT_READ | PROT_WRITE);
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, size_of::<T>(), MADV_HUGEPAGE);
    &*aligned.cast()
}

fn main() {
    let threads = thread::available_parallelism().map_or(12, |n| n.get());
    COUNTRIES.lock().unwrap().push("_UNMATCHED".into());
    let run = unsafe {
        Run {
            records: map_file("/data/records.csv"),
            regions: map_file("/data/regions.csv"),
            tables: zeroed_huge(),
            regions_done: Barrier::new(threads),
        }
    };
    thread::scope(|s| run.start(s, threads - 1));

    let totals = TOTALS.lock().unwrap();
    // `_UNMATCHED` sorts after every country code.
    let mut keys: Vec<_> = COUNTRIES.lock().unwrap().iter().cloned().zip(0..).collect();
    keys.sort_unstable();
    let mut out = String::new();
    for (key, group) in keys {
        let s = &totals[group];
        let row = &run.tables.seen[group * SEEN_WORDS..][..SEEN_WORDS];
        let distinct: u32 = row.iter().map(|w| w.load(Relaxed).count_ones()).sum();
        let [sum, min, max] = [s.amount_sum, BIAS - s.amount_lo, s.amount_hi - BIAS].map(money);
        let (count, id_sum) = (s.count, s.id_sum);
        let (name_bytes, note_bytes) = (s.lengths & 0xFFFF_FFFF, s.lengths >> 32);
        if count > 0 {
            out += &format!(
                "{key} {count} {id_sum} {sum} {min} {max} {distinct} {name_bytes} {note_bytes}\n"
            );
        }
    }
    std::fs::write("/out/aggregates.txt", out).unwrap();
}

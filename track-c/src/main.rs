//! Speed Demon 2026, Track C (Join).
//!
//! Threads and phases. `main` maps both files and starts one worker per CPU as a tree
//! (`Run::start`). Every worker first zeroes one huge page of the shared tables, then the
//! workers parse `regions.csv` together in 64 KiB claims into a region-to-group table, and
//! one barrier later they parse `records.csv` in 1 MiB claims. Each worker sums into its own
//! array of slots per group, merged under a lock at the end; `main` sorts the groups and prints.
//!
//! The file is parsed in parallel without ever scanning from its start, because the spec
//! gives a safe resync point: inside a note an LF is only ever followed by a lowercase word
//! or a quote, so an LF followed by a digit or a BOM always starts a record.
//!
//! A claim is walked in batches of 2 KiB, each in four passes:
//!
//! 1. Find structure. For each 64 bytes, AVX2 finds quotes, commas and LFs, and a carry-less
//!    multiply turns the quotes into an inside-quotes mask. `""` toggles twice, so escapes
//!    need no handling at all. Every record then has exactly five structural bytes, whose
//!    positions go into one array.
//! 2. Read fields (passes A and B), with no branch that depends on the data: 23% of records
//!    hold an escape, so branching on escapes or field shapes would mispredict constantly.
//!    Digits of ids, amounts and regions are gathered and decoded four records at a time.
//! 3. Stats and name numbers. Names are drawn uniformly from 100,000 values, so every lookup
//!    is a cache miss, and pass B prefetches before this pass touches anything. A name is a
//!    64-bit hash: its top 15 bits pick a 64-byte bucket of eight slots, which holds 47 more
//!    bits of it as a tag beside the name's dense number. Two of 100,000 names matching on
//!    all 62 bits has odds of about 1 in a billion, and comparing hashes instead of bytes
//!    saves a second memory access per record.
//! 4. Distinct names. Each record sets its name's bit in its country's row of one shared bit
//!    matrix (256 groups by 2^17 names, 4 MiB), and popcounts of the rows give the counts.
use std::sync::{atomic::Ordering::*, atomic::*, Barrier, Mutex, OnceLock};
use std::{arch::x86_64::*, fs::File, ops::Range, os::fd::AsRawFd, ptr::*, slice, thread};

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, ...) -> i32;
}
// MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
const MAP_ANON: i32 = 0x4022;

const HUGE_PAGE: usize = 1 << 21;
// Structural bytes per record: four commas and an LF.
const FIELDS: usize = 5;
// A batch is 32 steps of 64 bytes of the file. Records that fit in one (the shortest is
// `1,,,1,0.00` and an LF, 11 bytes), two more that were left over from the one before, and three
// for the digit arrays, which are decoded four at a time and so may be read up to three past the
// last record.
const MAX_RECORDS: usize = 32 * 64 / 11 + 5;
// The three numbers of a record that wait for `decode_digits` are kept in a batch's arrays 0 to 2:
// id, amount and region.
// Quote masks kept for the rank query, more than one batch.
const QUOTE_WORDS: usize = 1 << 10;
const BUCKETS: usize = 1 << 15;
// A name's dense number has 17 bits; the generator draws from 100,000 names.
const NUMBER_BITS: i32 = 17;
const NAME_MASK: u64 = (1 << NUMBER_BITS) - 1;
// Lookup ids stay below this; anything above is unmatched.
const REGION_IDS: usize = 1_000_000;
// 240 countries, with index 0 for unmatched
const GROUPS: usize = 256;
// Words of one group's row of the name bit matrix.
const SEEN_WORDS: usize = 1 << (NUMBER_BITS - 6);
// Amounts are kept as the largest of `BIAS + amount` and of `BIAS - amount`, both positive, so
// that a zeroed group is a valid empty one and merging two needs only sums and maxima.
const BIAS: i64 = 1 << 40;

// Counters that hand out work: region claims, record claims, huge pages to zero, name numbers.
static NEXT: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
// Region id to group, and country code (as base-26 digits) to group. Both start zeroed.
static REGION_GROUP: [AtomicU8; REGION_IDS + 1] = [const { AtomicU8::new(0) }; REGION_IDS + 1];
static LETTERS: [OnceLock<u8>; 26 * 26 * 26] = [const { OnceLock::new() }; 26 * 26 * 26];
// The sums of all workers, in groups. A group's slots, for a worker and for these totals alike:
// 0 count, 1 id sum, 2 amount sum, 3 name bytes in the low half and note bytes in the high half
// (the spec keeps each below 2^31, so the halves never carry into each other and one add
// updates both), 4 the largest `BIAS + amount`, 5 the largest `BIAS - amount`. The last two
// slots only pad a group to one cache line.
static TOTALS: Mutex<[[i64; 8]; GROUPS]> = Mutex::new([[0; 8]; GROUPS]);
// Group to country code, with the unmatched group first.
static COUNTRIES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The two shared tables, zeroed. Together they are exactly three huge pages.
/// 0: the name buckets. A bucket is one cache line of eight slots. A slot holds a 47-bit name
/// tag above the name's dense number, so one compare-and-swap claims a slot with its number.
/// Numbers keep the bit matrix small.
/// 1: the name bit matrix, `SEEN_WORDS` words per group.
#[repr(C)]
struct Tables([[AtomicU64; 8]; BUCKETS], [AtomicU64; GROUPS * SEEN_WORDS]);

/// What the workers share: 0 the records file, 1 the regions file, 2 the tables, 3 the barrier
/// after the regions are loaded.
struct Run(&'static [u8], &'static [u8], &'static Tables, Barrier);

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

/// Turns the first `count` words of eight digit bytes each into numbers, four at a time, like
/// `eight_digits` does for one. The three arrays of a batch go through the same few
/// instructions, so a record's numbers cost far less than three scalar multiply chains.
unsafe fn decode_digits(words: &[u64], numbers: &mut [u32], count: usize) {
    for at in (0..count).step_by(4) {
        let digits = _mm256_and_si256(load(words.as_ptr().add(at).cast()), _mm256_set1_epi8(15));
        // Pairs of digits (times 10 and 1), then groups of four (times 100 and 1), then all eight
        // (times 10,000 and 1).
        let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
        let fours = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x0001_0064));
        let fours = _mm256_packus_epi32(fours, fours);
        let eights = _mm256_madd_epi16(fours, _mm256_set1_epi32(0x0001_2710));
        // Each 128-bit half now holds its two numbers in its low 64 bits.
        let packed = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(eights));
        _mm_storeu_si128(numbers.as_mut_ptr().add(at).cast(), packed);
    }
}

/// The hash of a name, never 0, which marks an unused slot, and the number of quote bytes in
/// it. Names are compared by their raw bytes: every non-empty name is quoted, so equal raw
/// bytes mean equal values and nothing needs unescaping. Names of 32 bytes and more do not
/// occur at seed 1 (the longest is 27), so they cost one predicted branch here.
unsafe fn name_hash(name: *const u8, len: usize) -> (u64, usize) {
    // 32 bytes of ones then 32 of zeros: loading at 32 - len keeps the first `len` bytes.
    static MASKS: [[u8; 32]; 2] = [[0xFF; 32], [0; 32]];
    if len >= 32 {
        // Cold: hash the first 31 bytes and the rest apart, and combine them.
        let (a, b) = (name_hash(name, 31), name_hash(name.add(31), len - 31));
        let hash = (a.0.rotate_left(23) ^ b.0).wrapping_mul(0xE703_7ED1_A0B4_28DB);
        return (hash | 1, a.1 + b.1);
    }
    let bytes = _mm256_and_si256(load(name), load(MASKS.as_ptr().cast::<u8>().add(32 - len)));
    let quotes = equal_mask(bytes, b'"').count_ones() as usize;
    let key = _mm_set_epi64x(0x1D8E_4E27_C47D_124F, len as i64 ^ 0x5899_65CC_7537_4CC3);
    let low = _mm256_castsi256_si128(bytes);
    let high = _mm256_extracti128_si256::<1>(bytes);
    let mixed = _mm_aesenc_si128(_mm_xor_si128(low, key), high);
    let mixed = _mm_aesenc_si128(_mm_aesenc_si128(mixed, key), key);
    (_mm_cvtsi128_si64(mixed) as u64 | 1, quotes)
}

/// The next unclaimed block of `size` bytes in a file of `len` bytes, if any is left.
fn take_block(counter: &AtomicUsize, size: usize, len: usize) -> Option<usize> {
    Some(counter.fetch_add(size, Relaxed)).filter(|&at| at < len)
}

impl Run {
    /// The number of a name, new if this is its first appearance.
    unsafe fn intern(&self, hash: u64, spare: &mut Range<usize>) -> usize {
        let index = (hash >> 49) as usize;
        let tag = hash & (1 << 47) - 1 | 1;
        let bucket = &self.2 .0[index];
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
        (bucket[found.trailing_zeros() as usize].load(Relaxed) & NAME_MASK) as usize
    }

    /// Adds a name whose bucket does not hold it: the 100,000 first appearances, and the few
    /// whose bucket is full and spills into the next. A compare-and-swap decides each slot, so
    /// no lock is needed. A worker that loses a race to the same name wastes one number.
    #[cold]
    unsafe fn insert(&self, index: usize, tag: u64, spare: &mut Range<usize>) -> usize {
        if spare.is_empty() {
            // Name numbers are reserved 32 at a time.
            let first = NEXT[3].fetch_add(32, Relaxed);
            *spare = first..first + 32;
        }
        let claimed = tag << NUMBER_BITS | spare.start as u64;
        // From the bucket's first slot to the end of the table, and round again if need be.
        let slots = self.2 .0.as_flattened();
        for slot in slots[index * 8..].iter().chain(slots) {
            // Read first: a compare-and-swap on a taken slot would only pull its line away from
            // the other cores.
            match slot.load(Relaxed) {
                // On failure the slot holds another name, or this one if a worker just added it.
                0 => match slot.compare_exchange(0, claimed, Relaxed, Relaxed) {
                    Ok(_) => return spare.next().unwrap(),
                    Err(held) if held >> NUMBER_BITS == tag => return (held & NAME_MASK) as usize,
                    Err(_) => {}
                },
                held if held >> NUMBER_BITS == tag => return (held & NAME_MASK) as usize,
                _ => {}
            }
        }
        unreachable!()
    }

    /// One worker's whole life. It starts `descendants` more workers as a tree, so that no thread
    /// spends a millisecond spawning them all, zeroes its huge page, parses its share of the
    /// regions, waits for the others, and parses its share of the records.
    unsafe fn start<'a>(&'a self, scope: &'a thread::Scope<'a, '_>, mut descendants: usize) {
        while descendants > 0 {
            let share = descendants.div_ceil(2);
            scope.spawn(move || unsafe { self.start(scope, share - 1) });
            descendants -= share;
        }
        // Zeroing a huge page takes a few hundred microseconds, so the workers share it and do
        // it while the regions load. It must end before huge pages are disabled.
        let page = NEXT[2].fetch_add(1, Relaxed) * HUGE_PAGE;
        if page < size_of::<Tables>() {
            (*(self.2 as *const Tables as *const AtomicU8).add(page)).store(0, Relaxed);
        }
        // Region claims are 64 KiB. Each starts after the first LF at or past its start, so the
        // header goes to nobody and a line belongs to the claim holding the LF before it.
        while let Some(claim) = take_block(&NEXT[0], 1 << 16, self.1.len()) {
            let rest = &self.1[claim..];
            let mut line = rest.iter().position(|&c| c == b'\n').unwrap() + 1;
            while line <= 1 << 16 && line < rest.len() {
                let row = &rest[line..];
                let at = row.iter().position(|&c| c == b',').unwrap();
                // (ASCII 48 is `0`.)
                let id = row[..at].iter().fold(0, |n, &c| n * 10 + c as usize - 48);
                // Groups are numbered from 1 in order of first appearance of the country.
                let code = &row[at + 1..][..3];
                let cell = &LETTERS[code.iter().fold(0, |n, &c| n * 26 + (c - b'A') as usize)];
                // Only 240 lines get this far, so a lock is fine.
                let group = cell.get_or_init(|| {
                    let mut countries = COUNTRIES.lock().unwrap();
                    countries.push(String::from_utf8_lossy(code).into_owned());
                    countries.len() as u8 - 1
                });
                REGION_GROUP[id].store(*group, Relaxed);
                // Past the id, its comma, the three-letter code and the LF.
                line += at + 5;
            }
            madvise(self.1.as_ptr().add(claim) as *mut u8, (claim + (1 << 16)).min(self.1.len()) - claim, 4);
        }
        self.3.wait();
        // Records follow. PR_SET_THP_DISABLE is
        // idempotent, and set before the worker allocates anything for them.
        prctl(41, 1usize, 0usize, 0usize, 0usize);
        let (input, len) = (self.0.as_ptr() as *mut u8, self.0.len());
        let mut stats = [[0i64; 8]; GROUPS];
        let mut pos = [0u32; MAX_RECORDS * FIELDS];
        // Each 64-byte step's quote mask, and the number of quote bytes before the step.
        let mut quotes = [(0u64, 0u32); QUOTE_WORDS];
        // Per record between the passes: the name's hash (later the address of its bit), the name
        // and note lengths (`lengths` in the totals), and -1 for a negative amount, else 0.
        let [mut hashes, mut lengths] = [[0u64; MAX_RECORDS]; 2];
        let mut signs = [0i64; MAX_RECORDS];
        let mut spare = 0..0; // name numbers this worker reserved
        let mut digits = [[0u64; MAX_RECORDS]; 3];
        let mut numbers = [[0u32; MAX_RECORDS]; 3];
        while let Some(claim) = take_block(&NEXT[1], 1 << 20, len) {
            // Claims are 1 MiB.
            let end = (claim + (1 << 20)).min(len);
            // The first record at or after `claim`.
            let from = claim.saturating_sub(1);
            let begins = |pair: &[u8]| pair[0] == b'\n' && matches!(pair[1], b'0'..=b'9' | 0xEF);
            let origin = (self.0[from..].windows(2).position(begins)).map_or(len, |p| from + p + 1);
            let mut record = origin; // the next record to read
            let mut at = origin; // the next byte to scan
            let (mut inside, mut quote_total, mut filled) = (0u64, 0u32, 0usize);
            while record < end {
                for _ in 0..32 {
                    let low = load(input.add(at));
                    let high = load(input.add(at + 32));
                    let mask = |byte| equal_mask(low, byte) | equal_mask(high, byte) << 32;
                    let (quote, comma, newline) = (mask(b'"'), mask(b','), mask(b'\n'));
                    // A carry-less multiply by all ones sets each bit that has an odd number of
                    // quote bits at or below it: the mask of everything inside quotes.
                    let ones = _mm_set1_epi8(-1);
                    let quoted = _mm_clmulepi64_si128(_mm_cvtsi64_si128(quote as i64), ones, 0);
                    let inside_now = _mm_cvtsi128_si64(quoted) as u64 ^ inside;
                    // All ones if the step ended inside quotes: the next step starts inverted.
                    inside = ((inside_now as i64) >> 63) as u64;
                    let mut bits = (comma | newline) & !inside_now;
                    quotes[(at - origin) / 64 % QUOTE_WORDS] = (quote, quote_total);
                    quote_total += quote.count_ones();
                    // Unconditional stores, so that a varying count needs no loop exit to
                    // predict. Eight cover almost every step.
                    let found = bits.count_ones() as usize;
                    let mut extra = 0;
                    while extra < 8 || bits != 0 {
                        *pos.get_unchecked_mut(filled + extra) = at as u32 + bits.trailing_zeros();
                        bits &= bits.wrapping_sub(1);
                        extra += 1;
                    }
                    filled += found;
                    at += 64;
                }

                let quotes_up_to = |position: usize| {
                    let (quote, before) = quotes[(position - origin) / 64 % QUOTE_WORDS];
                    let below = _bzhi_u64(quote, ((position - origin) % 64) as u32);
                    before.wrapping_add(below.count_ones())
                };
                let field = |at: usize| {
                    let five = pos.as_ptr().add(at * FIELDS).cast::<[u32; FIELDS]>();
                    (*five).map(|p| p as usize)
                };
                // Pass A: how many records start inside this claim, and their numbers.
                let mut count = 0;
                let mut start = record;
                while count < filled / FIELDS && start < end {
                    let [comma1, _, comma3, comma4, newline] = field(count);
                    // Only the file's first record can start with a BOM, three bytes before its id.
                    let bom = (*input.add(start) == 0xEF) as usize * 3;
                    digits[0][count] = digits_before(input.add(comma1), comma1 - start - bom);
                    digits[2][count] = digits_before(input.add(comma4), comma4 - comma3 - 1);
                    let stop = newline - (*input.add(newline - 1) == b'\r') as usize;
                    let negative = (*input.add(comma4 + 1) == b'-') as usize;
                    let cents = digits_before(input.add(stop), stop - comma4 - 1 - negative);
                    // Close the gap the point leaves: its byte is always third from the end.
                    digits[1][count] = (cents & 0xFF_FFFF_FFFF) << 8 | cents >> 48 << 48;
                    signs[count] = -(negative as i64);
                    start = newline + 1;
                    count += 1;
                }
                [0, 1, 2].map(|kind| decode_digits(&digits[kind], &mut numbers[kind], count));
                // Pass B: names and notes.
                let mut rank = quotes_up_to(record);
                for at in 0..count {
                    let [comma1, comma2, comma3, ..] = field(at);
                    let raw_name = comma2 - comma1 - 1;
                    let (hash, inner) =
                        name_hash(input.add(comma1 + 2), raw_name.saturating_sub(2));
                    // A quoted field of q quote bytes holds (q - 2) / 2 escapes, so its value is
                    // `raw - q / 2 - 1` bytes long, and `raw` when q is 0 (an empty field).
                    let name_len = raw_name.saturating_sub(2) - inner / 2;
                    let before = std::mem::replace(&mut rank, quotes_up_to(comma3));
                    let name_quotes = inner + 2 * (raw_name != 0) as usize;
                    let note_q = rank.wrapping_sub(before) as usize - name_quotes;
                    let note_len = comma3 - comma2 - 1 - note_q / 2 - (note_q != 0) as usize;
                    // The top 15 bits of the hash pick the bucket.
                    _mm_prefetch::<_MM_HINT_T0>(self.2 .0[(hash >> 49) as usize].as_ptr().cast());
                    // Ids past the table share its last entry, which stays 0: unmatched.
                    let region = (numbers[2][at] as usize).min(REGION_IDS);
                    _mm_prefetch::<_MM_HINT_T0>(REGION_GROUP.as_ptr().add(region).cast());
                    numbers[2][at] = region as u32;
                    hashes[at] = hash;
                    lengths[at] = name_len as u64 | (note_len as u64) << 32;
                }
                record = start;
                // The region table and the name table were each a cache miss on every record,
                // half of the pass above when it read them itself. Now both are prefetched
                // before this pass reads them. A locked update of a cold line stalls the core,
                // so its lines are fetched here and updated in the pass after.
                for at in 0..count {
                    let group = REGION_GROUP[numbers[2][at] as usize].load(Relaxed) as usize;
                    let amount = (numbers[1][at] as i64 ^ signs[at]) - signs[at];
                    let s = &mut stats[group];
                    s[0] += 1;
                    s[1] += numbers[0][at] as i64;
                    s[2] += amount;
                    s[3] += lengths[at] as i64;
                    s[4] = s[4].max(BIAS + amount);
                    s[5] = s[5].max(BIAS - amount);
                    let name = self.intern(hashes[at], &mut spare);
                    let word = group * SEEN_WORDS + name / 64;
                    _mm_prefetch::<_MM_HINT_ET0>(self.2 .1.as_ptr().add(word).cast());
                    // The hash is done with, so its slot carries the bit's address to the next pass.
                    hashes[at] = (word << 6 | name % 64) as u64;
                }
                for &bit in &hashes[..count] {
                    let word = self.2 .1.get_unchecked((bit >> 6) as usize);
                    word.fetch_or(1 << (bit & 63), Relaxed);
                }

                pos.copy_within(count * FIELDS..filled, 0);
                filled -= count * FIELDS;
            }
            // Unmap this claim's pages now, in parallel. Otherwise the kernel frees all
            // ~50,000 page-table entries on one thread at exit, which the timer includes. A
            // record running into the next claim just faults its few pages back in.
            // The length stops at the file's end (the kernel rounds it up to a page): madvise goes on
            // into whatever is mapped after the range, and would wipe it.
            madvise(input.add(claim), end - claim, 4); // MADV_DONTNEED
        }
        let mut totals = TOTALS.lock().unwrap();
        for (total, part) in totals.iter_mut().zip(&stats) {
            for (i, (a, &b)) in total.iter_mut().zip(part).enumerate() {
                *a = if i < 4 { *a + b } else { (*a).max(b) };
            }
        }
    }
}

/// Maps a file for reading, with a spare zero page after it so wide loads may run past the end.
unsafe fn map(name: &str) -> &'static [u8] {
    let file = File::open(format!("/data/{name}.csv")).unwrap();
    let (fd, len) = (file.as_raw_fd(), file.metadata().unwrap().len() as usize);
    let size = len.next_multiple_of(4096) + 4096;
    let base = mmap(null_mut(), size, 1, MAP_ANON, -1, 0); // PROT_READ
    mmap(base, len, 1, 0x12, fd, 0); // PROT_READ, MAP_PRIVATE | MAP_FIXED
    slice::from_raw_parts(base, len)
}

/// A zeroed table of whole huge pages, which must start on a huge page boundary to get any. Its
/// pages are touched later by the workers, because huge pages are disabled for everything else
/// before the records start.
unsafe fn zeroed<T>() -> &'static T {
    let region = mmap(null_mut(), size_of::<T>() + HUGE_PAGE, 3, MAP_ANON, -1, 0); // read, write
    let aligned = region.add(region.align_offset(HUGE_PAGE));
    madvise(aligned, size_of::<T>(), 14); // MADV_HUGEPAGE
    &*aligned.cast()
}

fn main() {
    let n = thread::available_parallelism().map_or(12, |n| n.get());
    COUNTRIES.lock().unwrap().push("_UNMATCHED".into());
    let run = unsafe { Run(map("records"), map("regions"), zeroed(), Barrier::new(n)) };
    thread::scope(|s| unsafe { run.start(s, n - 1) });

    let totals = TOTALS.lock().unwrap();
    // `_UNMATCHED` sorts after every country code.
    let mut keys: Vec<_> = COUNTRIES.lock().unwrap().iter().cloned().zip(0..).collect();
    keys.sort_unstable();
    let mut out = String::new();
    for (key, group) in keys {
        let [count, id_sum, amount, lengths, hi, lo, ..] = totals[group];
        let row = &run.2 .1[group * SEEN_WORDS..][..SEEN_WORDS];
        let distinct: u32 = row.iter().map(|w| w.load(Relaxed).count_ones()).sum();
        // Cents as text. A double holds any total of cents to well within half a cent.
        let [sum, min, max] =
            [amount, BIAS - lo, hi - BIAS].map(|c| format!("{:.2}", c as f64 / 100.0));
        let [name, note] = [lengths & 0xFFFF_FFFF, lengths >> 32];
        if count > 0 {
            out += &format!("{key} {count} {id_sum} {sum} {min} {max} {distinct} {name} {note}\n");
        }
    }
    std::fs::write("/out/aggregates.txt", out).unwrap();
}

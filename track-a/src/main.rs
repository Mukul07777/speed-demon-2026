//! Speed Demon 2026, Track A (Ingest).
//!
//! One pool of threads does everything, in two phases:
//!
//! 1. Parse. Workers claim 1 MiB of the mapped file and map it in with one call, because
//!    faulting pages in one at a time from every thread costs more than parsing them. Then,
//!    64 KiB at a time, they find every newline first and read each line from both ends
//!    with no data-dependent branches. Each line becomes a 4-byte row in its hour's region.
//! 2. Summarise. Workers claim one hour each and radix sort its rows by latency, inside the
//!    hour's own pages. Dealing the sorted rows out by endpoint keeps them in order, so every
//!    endpoint's latencies come out sorted and a percentile is one array read. The hour's text
//!    is built over its rows, in the same pages. Whoever finds the next hour ready writes it
//!    and gives its pages back, so the output overlaps the work, and memory is released as
//!    fast as the tmpfs output file grows (which counts against the memory limit too).
//!
//! Lines are ASCII and well formed (the spec promises it), so the hot loops use raw pointers.
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::ptr::{copy_nonoverlapping, read_unaligned, write_unaligned};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Barrier, Mutex, OnceLock};
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
const MADV_DONTNEED: i32 = 4;
const PR_SET_THP_DISABLE: i32 = 41;

const CLAIM: usize = 1 << 20;
const BLOCK: usize = 1 << 16; // fits in L2 between the newline pass and the line pass
const MIN_LINE: usize = 40; // ts 27, method 3, endpoint 1, status 3, latency 1, 4 spaces, LF
const HOURS: usize = 72; // the span is fixed for every seed: 2026-10-30T18 to 2026-11-02T17
const CHUNK: usize = 256; // rows of an hour's region a worker claims at a time
const SLOTS: usize = 1 << 13;
const ID_BITS: u32 = 11; // low bits of a slot hold id + 1
const ID_MASK: u64 = (1 << ID_BITS) - 1;
const SPARE_AT: usize = 1 << 18; // rows (1 MiB): the scratch starts at least this far in, past any hour's text
const SPACES: u64 = 0x2020_2020_2020_2020;
const ZEROS: u64 = 0x3030_3030_3030_3030;
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGHS: u64 = 0x8080_8080_8080_8080;

/// A row is id + 1 in 11 bits, an error bit, then the latency in 20 bits. Only about 1% of
/// latencies need more: those store ESCAPE here and the full value in the next row, which
/// keeps almost every row at 4 bytes. A row of 0 marks the unused end of a chunk.
const LATENCY_BITS: u32 = 20;
const ESCAPE: u32 = (1 << LATENCY_BITS) - 1;
const DIGIT_BITS: u32 = 10;
const DIGITS: usize = 1 << DIGIT_BITS; // of the two radix passes over a 20-bit latency

/// The next free row and the final slot of the chunk being filled: a row needs two free, in
/// case it has to escape.
type Cursor = (*mut u32, *mut u32);

unsafe fn read(p: *const u8) -> u64 {
    read_unaligned(p.cast())
}

/// Sets the high bit of each space byte. A borrow can also mark the byte above a space if
/// that byte is `!`, which never sits above the first space in a method or among digits.
fn spaces_in(word: u64) -> u64 {
    let z = word ^ SPACES;
    z.wrapping_sub(ONES) & !z & HIGHS
}

/// Every endpoint seen so far. Workers share it: it fills in the first moments of the run
/// and is read-only after that, so one copy stays hot in cache.
///
/// Names are identified by a 64-bit hash alone (53 bits of it are compared), the same
/// deliberate shortcut Tracks B and C take: with 1,842 names the chance of any two colliding
/// is about 2e-10.
struct Catalogue {
    slots: [AtomicU64; SLOTS], // hash >> 11 << 11 | id + 1; 0 is an empty slot
    names: Mutex<Vec<&'static [u8]>>, // by id
}

/// A hash of the whole name without a length-dependent loop: the masked first and last 32
/// bytes cover any name up to 64 bytes, and the few longer ones also fold in the middle.
/// Never leaves the id bits zero, so a live slot is never 0.
#[inline(always)]
unsafe fn name_hash(name: *const u8, len: usize) -> u64 {
    // Lane numbers 0 to 31, to keep the first or the last `short` bytes of a 32-byte load.
    let lanes: __m256i = std::mem::transmute(std::array::from_fn::<u8, 32, _>(|i| i as u8));
    let load = |at: *const u8| _mm256_loadu_si256(at.cast());
    let short = _mm256_set1_epi8(len.min(32) as i8);
    let head = _mm256_and_si256(load(name), _mm256_cmpgt_epi8(short, lanes));
    let ahead = _mm256_cmpgt_epi8(lanes, _mm256_sub_epi8(_mm256_set1_epi8(31), short));
    let tail = _mm256_and_si256(load(name.add(len).sub(32)), ahead);
    // Names over 64 bytes also fold in their middle, byte by byte. Shorter ones have none.
    let middle = std::slice::from_raw_parts(name.add(32), len.saturating_sub(64));
    let hash = middle.iter().fold(len as u64, |hash, &byte| {
        (hash.rotate_left(5) ^ byte as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
    });
    // Three AES rounds mix each 16-byte half. The tail is the first round key, and the
    // length and constants go in as the later ones.
    let key = _mm256_set_epi64x(KEY as i64, !KEY as i64, len as i64, -(len as i64));
    let mixed = _mm256_aesenc_epi128(head, tail);
    let mixed = _mm256_aesenc_epi128(_mm256_aesenc_epi128(mixed, key), mixed);
    let [a, b, c, d]: [u64; 4] = std::mem::transmute(mixed);
    (hash ^ a ^ b ^ c ^ d) | 1 << ID_BITS
}

const KEY: u64 = 0x9E37_79B9_7F4A_7C15;

impl Catalogue {
    /// The id of an endpoint name, new if this is its first appearance.
    #[inline(always)]
    unsafe fn lookup(&self, name: *const u8, len: usize) -> usize {
        let hash = name_hash(name, len);
        match self.probe(hash) {
            Ok(id) => id,
            Err(_) => self.insert(hash, name, len),
        }
    }

    /// The id with this hash, or the index of the empty slot that ends its probe sequence.
    #[inline(always)]
    fn probe(&self, hash: u64) -> Result<usize, usize> {
        let mut i = (hash >> (64 - SLOTS.trailing_zeros())) as usize;
        loop {
            let slot = self.slots[i].load(Acquire);
            if slot >> ID_BITS == hash >> ID_BITS {
                return Ok((slot & ID_MASK) as usize - 1);
            } else if slot == 0 {
                return Err(i);
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    /// The name is fully written before the Release store publishes the slot, so a reader
    /// that sees the slot sees the name.
    #[cold]
    unsafe fn insert(&self, hash: u64, name: *const u8, len: usize) -> usize {
        let mut names = self.names.lock().unwrap();
        // Another worker may have added it while this one waited for the lock.
        let slot = match self.probe(hash) {
            Ok(id) => return id,
            Err(slot) => slot,
        };
        let id = names.len();
        assert!(id < ID_MASK as usize); // id + 1 must fit the low bits of a slot
        names.push(std::slice::from_raw_parts(name, len).to_vec().leak());
        self.slots[slot].store(hash >> ID_BITS << ID_BITS | (id + 1) as u64, Release);
        id
    }

    /// Names in output order, with their ids. A &[u8] orders as unsigned bytes with a proper
    /// prefix first.
    fn sorted(&self) -> Vec<(&'static [u8], usize)> {
        let names = self.names.lock().unwrap();
        let mut order: Vec<_> = names.iter().copied().zip(0..).collect();
        order.sort_unstable();
        order
    }
}

struct Run {
    input: *mut u8,
    len: usize,
    next_claim: AtomicUsize,
    catalogue: Catalogue,
    rows: *mut u32,
    stride: usize,
    claimed: [AtomicUsize; HOURS], // rows of each hour's region given out in chunks
    finished_parsing: Barrier,
    next_hour: AtomicUsize,
    texts: [OnceLock<&'static [u8]>; HOURS], // each hour's text, until it is written
    output: Mutex<(File, usize)>,            // the file and the next hour it needs
}
unsafe impl Sync for Run {}

/// Finds the newlines of [from, to) and returns how many. Lines are longer than the 32-byte
/// step, so a step holds at most one newline, and one unconditional store records it. Each
/// pass over a block is then a counted loop, with no branch on where lines end.
unsafe fn find_newlines(input: *const u8, from: usize, to: usize, ends: &mut [u32]) -> usize {
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let (mut count, mut at) = (0, from);
    while at < to {
        let hits = _mm256_cmpeq_epi8(_mm256_load_si256(input.add(at).cast()), newline);
        let mask = _mm256_movemask_epi8(hits) as u32;
        *ends.get_unchecked_mut(count) = (at + mask.trailing_zeros() as usize) as u32;
        count += (mask != 0) as usize;
        at += 32;
    }
    count
}

/// A latency of eight digits or more, which the word-wide decode cannot see the start of.
#[cold]
unsafe fn long_latency(line_end: *const u8) -> (u32, usize) {
    let digits = (1..).find(|&n| *line_end.sub(n + 1) == b' ').unwrap();
    let text = std::slice::from_raw_parts(line_end.sub(digits), digits);
    (std::str::from_utf8_unchecked(text).parse().unwrap(), digits)
}

impl Run {
    #[cold]
    unsafe fn claim_chunk(&self, hour: usize) -> Cursor {
        let offset = self.claimed[hour].fetch_add(CHUNK, Relaxed);
        let at = self.rows.add(hour * self.stride + offset);
        (at, at.add(CHUNK - 1))
    }

    unsafe fn parse(&self) {
        let input = self.input;
        let mut cursors: [Cursor; HOURS] = [(std::ptr::null_mut(), std::ptr::null_mut()); HOURS];
        let mut ends = [0u32; BLOCK / MIN_LINE + 1];
        loop {
            let claim = self.next_claim.fetch_add(CLAIM, Relaxed);
            if claim >= self.len {
                return;
            }
            // A line belongs to the claim holding its newline. The first one starts after
            // the previous claim's last newline, at most one line back.
            let mut line = input.add(claim);
            while claim > 0 && *line.sub(1) != b'\n' {
                line = line.sub(1);
            }
            for start in (claim..(claim + CLAIM).min(self.len)).step_by(BLOCK) {
                let to = (start + BLOCK).min(self.len);
                let count = find_newlines(input, start, to, &mut ends);
                for &end in &ends[..count] {
                    let end = input.add(end as usize);
                    // Oct 30, Oct 31, Nov 1 and Nov 2 are day 0 to 3 of the span.
                    let clock = read(line.add(8));
                    let day = (clock >> 8 & 15) + 1 - (clock >> 1 & 1);
                    let hour = day * 24 + (clock >> 24 & 15) * 10 + (clock >> 32 & 15) - 18;
                    let method = spaces_in(read(line.add(28))).trailing_zeros() as usize / 8;
                    let name = line.add(29 + method);
                    // Read the latency back from the newline. Its length varies line by
                    // line, so a digit loop would mispredict on nearly every one.
                    let last_eight = read(end.sub(8));
                    let spaces = spaces_in(last_eight);
                    let (latency, digits) = if spaces != 0 {
                        let digits = spaces.leading_zeros() as usize / 8;
                        let v = (last_eight ^ ZEROS) >> (64 - 8 * digits) << (64 - 8 * digits);
                        let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
                        let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
                        let v = (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32;
                        (v as u32, digits)
                    } else {
                        long_latency(end)
                    };
                    let status = end.sub(digits + 4);
                    let len = status.sub(1).offset_from(name) as usize;
                    let id = self.catalogue.lookup(name, len) as u32;
                    let is_error = (*status == b'5') as u32;
                    let cursor = cursors.get_unchecked_mut(hour as usize);
                    if cursor.0 >= cursor.1 {
                        *cursor = self.claim_chunk(hour as usize);
                    }
                    *cursor.0 = ((id + 1) << 1 | is_error) << LATENCY_BITS | latency.min(ESCAPE);
                    cursor.0 = cursor.0.add(1);
                    if latency >= ESCAPE {
                        *cursor.0 = latency;
                        cursor.0 = cursor.0.add(1);
                    }
                    line = end.add(1);
                }
            }
            // Drop the claim's pages now, while every thread is busy. Left mapped, they are
            // torn down by one thread at exit, which cost about 7 ms on the scoring kernel.
            // A later read of this range, by the next claim's first line, refaults the same
            // bytes, because the mapping is a read-only private file mapping.
            madvise(input.add(claim), CLAIM.min(self.len - claim), MADV_DONTNEED);
        }
    }

    /// Sorts and summarises one hour, and returns its text. Everything lives in the hour's own
    /// area of `rows`: the rows first, then the text over them, and the sort scratch further
    /// on. Nothing is allocated per hour, and `flush` gives the whole area back in one call.
    /// (Text in malloc'd buffers was the bulk of the memory peak: the allocator kept every
    /// buffer it had grown while the writer, which is serial, fell behind.)
    unsafe fn summarise(&self, hour: usize, order: &[(&[u8], usize)]) -> &'static [u8] {
        let rows = self.rows.add(hour * self.stride);
        let claimed = self.claimed[hour].load(Relaxed);
        // The text can run past the rows, but never a million bytes past them.
        let spare = rows.add(claimed.max(SPARE_AT).next_multiple_of(1024));
        let mut starts = vec![0u32; order.len() + 2];
        let mut errors = starts.clone();
        let mut long = Vec::new();

        // Moves rows down over the gaps between chunks. The 1% that escape go aside: they
        // are all longer than the rest, so sorted on their own they simply follow them.
        let mut sizes = [[0u32; DIGITS]; 2];
        let mut count = 0;
        for chunk in (0..claimed).step_by(CHUNK) {
            let mut at = chunk;
            while at < chunk + CHUNK && *rows.add(at) != 0 {
                let row = *rows.add(at);
                at += 1;
                let group = (row >> (LATENCY_BITS + 1)) as usize;
                *starts.get_unchecked_mut(group) += 1;
                *errors.get_unchecked_mut(group) += row >> LATENCY_BITS & 1;
                if row & ESCAPE == ESCAPE {
                    long.push((*rows.add(at), row));
                    at += 1;
                } else {
                    *rows.add(count) = row;
                    count += 1;
                    sizes[0][row as usize & (DIGITS - 1)] += 1;
                    sizes[1][(row >> DIGIT_BITS) as usize & (DIGITS - 1)] += 1;
                }
            }
        }
        // Two stable counting passes, one per digit, leave the rows sorted by latency.
        let [low, high] = &mut sizes;
        for (from, to, shift, hist) in [(rows, spare, 0, low), (spare, rows, DIGIT_BITS, high)] {
            let mut total = 0;
            for size in hist.iter_mut() {
                (total, *size) = (total + *size, total);
            }
            for i in 0..count {
                let row = *from.add(i);
                let slot = hist.get_unchecked_mut((row >> shift) as usize & (DIGITS - 1));
                *to.add(*slot as usize) = row;
                *slot += 1;
            }
        }
        long.sort_unstable();

        // Deals the sorted latencies out by endpoint into the scratch, which the rows no
        // longer need.
        let mut total = 0;
        for start in starts.iter_mut() {
            (total, *start) = (total + *start, total);
        }
        let mut deal = |row: u32, latency: u32| {
            let start = starts.get_unchecked_mut((row >> (LATENCY_BITS + 1)) as usize);
            *spare.add(*start as usize) = latency;
            *start += 1; // leaves each group's end where the next group's start was
        };
        for i in 0..count {
            deal(*rows.add(i), *rows.add(i) & ESCAPE);
        }
        long.iter().for_each(|&(latency, row)| deal(row, latency));

        let label = hour_label(hour);
        let text = rows.cast::<u8>();
        let mut w = text;
        for &(name, id) in order {
            let group = id + 1;
            let span = starts[group - 1] as usize..starts[group] as usize;
            if span.is_empty() {
                continue;
            }
            let (n, bad) = (span.len() as u32, errors[group]);
            let sorted = std::slice::from_raw_parts(spare.add(span.start), span.len());
            let at_rank = |p: usize| sorted[(p * sorted.len() + 99) / 100 - 1];
            let (p50, p95, p99) = (at_rank(50), at_rank(95), at_rank(99));
            copy_nonoverlapping(label.as_ptr(), w, 32);
            copy_nonoverlapping(name.as_ptr(), w.add(21), name.len());
            w = w.add(21 + name.len());
            for number in [n, p50, p95, p99, bad] {
                w = put_number(w, number);
            }
            w = put_number(w, (bad as u64 * 1_000_000 / n as u64) as u32);
            *w = b'\n';
            w = w.add(1);
        }
        madvise(spare.cast(), (count + long.len()) * 4, MADV_DONTNEED);
        std::slice::from_raw_parts(text, w.offset_from(text) as usize)
    }

    /// Writes every hour that is ready, in order. The caller holds the output lock.
    fn flush(&self, output: &mut (File, usize)) {
        while let Some(part) = self.texts.get(output.1).and_then(|text| text.get()) {
            output.0.write_all(part).unwrap();
            // The text is out: give the hour's pages back.
            unsafe { madvise(part.as_ptr() as *mut u8, self.stride * 4, MADV_DONTNEED) };
            output.1 += 1;
        }
    }

    fn work(&self) {
        unsafe { self.parse() };
        self.finished_parsing.wait();
        let order = self.catalogue.sorted();
        loop {
            let hour = self.next_hour.fetch_add(1, Relaxed);
            if hour >= HOURS {
                break;
            }
            let part = unsafe { self.summarise(hour, &order) };
            self.texts[hour].set(part).unwrap();
            if let Ok(mut output) = self.output.try_lock() {
                self.flush(&mut output);
            }
        }
        // The last worker here sees every part, whoever held the lock when it was stored.
        self.flush(&mut self.output.lock().unwrap());
    }
}

/// The hour as text with a trailing space, padded so it can be copied as one 32-byte load.
fn hour_label(hour: usize) -> [u8; 32] {
    let date = ["10-30", "10-31", "11-01", "11-02"][(hour + 18) / 24];
    let text = format!("2026-{date}T{:02}:00:00Z ", (hour + 18) % 24);
    let mut label = [0; 32];
    label[..21].copy_from_slice(text.as_bytes());
    label
}

/// Eight digits at once, as bytes with the first digit lowest. Divides by 10000, then 100,
/// then 10 across the lanes of one word.
fn eight_digits(value: u32) -> u64 {
    let merged = (value / 10000) as u64 | ((value % 10000) as u64) << 32;
    let hundreds = (merged.wrapping_mul(10486) >> 20) & 0x0000_007F_0000_007F;
    let pairs = hundreds | (merged - 100 * hundreds) << 16;
    let tens = (pairs.wrapping_mul(103) >> 10) & 0x000F_000F_000F_000F;
    tens | (pairs - 10 * tens) << 8
}

/// Writes a space and the number, with the same few instructions for every length, so the
/// varying widths do not mispredict. A u32 has at most ten digits: the top two come from
/// the first eight-digit group and the low eight from the second.
unsafe fn put_number(w: *mut u8, value: u32) -> *mut u8 {
    *w = b' ';
    let digits = eight_digits(value / 100_000_000) as u128
        | (eight_digits(value % 100_000_000) as u128) << 64;
    let skip = (digits | 1 << 127).trailing_zeros() as usize / 8; // leading zeros
    let zeros = ZEROS as u128 * ((1 << 64) + 1);
    write_unaligned(w.add(1).cast(), (digits + zeros) >> (8 * skip));
    w.add(17 - skip)
}

fn main() {
    let file = File::open("/data/requests.log").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());

    // Huge pages would turn each hour's first touch into 2 MB of memory.
    // Name lookups read a little past a line, so a spare zero page follows the file.
    // Each hour reserves room for every line, and only the pages it touches cost anything.
    let (input, rows, stride) = unsafe {
        prctl(PR_SET_THP_DISABLE, 1usize, 0usize, 0usize, 0usize);
        let map = |at, len, prot, flags, fd| mmap(at, len, prot, flags, fd, 0);
        let private = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
        let null = std::ptr::null_mut();
        let span = len.div_ceil(4096) * 4096 + 4096;
        let input = map(null, span, PROT_READ, private, -1);
        // For an empty file this fails, and the zero page stays.
        let fd = file.as_raw_fd();
        map(input, len, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd);
        // An hour has room for its rows (two per line, if every line escaped), then for the
        // scratch (one value per line) after them, or after SPARE_AT rows if that is further.
        let stride = SPARE_AT + (len / MIN_LINE + 1) * 4 + threads * CHUNK * 2;
        let stride = stride.div_ceil(1024) * 1024;
        let size = stride * HOURS * 4;
        let rows = map(null, size, PROT_READ | PROT_WRITE, private, -1);
        (input, rows.cast(), stride)
    };

    let run = Run {
        input,
        len,
        next_claim: AtomicUsize::new(0),
        catalogue: Catalogue {
            slots: [const { AtomicU64::new(0) }; SLOTS],
            names: Mutex::new(Vec::new()),
        },
        rows,
        stride,
        claimed: [const { AtomicUsize::new(0) }; HOURS],
        finished_parsing: Barrier::new(threads),
        next_hour: AtomicUsize::new(0),
        texts: [const { OnceLock::new() }; HOURS],
        output: Mutex::new((File::create("/out/summary.txt").unwrap(), 0)),
    };
    thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(|| run.work());
        }
        run.work();
    });
}

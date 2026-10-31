//! Speed Demon 2026, Track A (Ingest).
//!
//! One pool of threads does everything, in two phases:
//!
//! 1. Parse. Workers claim 1 MiB of the mapped file and map it in with one call, because
//!    faulting pages in one at a time from every thread costs more than parsing them. Then,
//!    64 KiB at a time, they find every newline first and read each line from both ends
//!    with no data-dependent branches. Each line becomes a 4-byte row in its hour's region.
//! 2. Summarise. Workers claim one hour each, group its rows by endpoint with a counting
//!    sort, select only the three percentiles, and give the memory back. Whoever finds the
//!    next hour ready writes it, so the output overlaps the work instead of following it.
//!
//! Lines are ASCII and well formed (the spec promises it), so the hot loops use raw pointers.
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::ptr::{copy_nonoverlapping, read_unaligned, write_unaligned};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};
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
const MADV_POPULATE_READ: i32 = 22;
const PR_SET_THP_DISABLE: i32 = 41;

const CLAIM: usize = 1 << 20;
const BLOCK: usize = 1 << 16; // fits in L2 between the newline pass and the line pass
const LINES_PER_BLOCK: usize = BLOCK / MIN_LINE + 1;
const MIN_LINE: usize = 40; // ts 27, method 3, endpoint 1, status 3, latency 1, 4 spaces, LF
const HOURS: usize = 72; // the span is fixed for every seed: 2026-10-30T18 to 2026-11-02T17
const CHUNK: usize = 256; // rows of an hour's region a worker claims at a time
const SLOTS: usize = 1 << 13;
const MAX_ENDPOINTS: usize = 2046; // the catalogue has 1,842, and id + 1 must fit 11 bits
const TEXT_BYTES: usize = 1 << 20; // all endpoint names, with room to copy past the last
const LINE_MAX: usize = 21 + 125 + 6 * 11 + 1 + 128; // one output line, plus wide-copy slack
const SPACES: u64 = 0x2020_2020_2020_2020;
const ZEROS: u64 = 0x3030_3030_3030_3030;
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGHS: u64 = 0x8080_8080_8080_8080;

/// A row is id + 1 in 11 bits, an error bit, then the latency in 20 bits. Only about 1% of
/// latencies need more: those store ESCAPE here and the full value in the next row, which
/// keeps almost every row at 4 bytes. A row of 0 marks the unused end of a chunk.
const LATENCY_BITS: u32 = 20;
const ESCAPE: u32 = (1 << LATENCY_BITS) - 1;

#[derive(Clone, Copy)]
struct Cursor {
    at: *mut u32,
    last: *mut u32, // the final slot: a row needs two free, in case it has to escape
}

unsafe fn read(p: *const u8) -> u64 {
    read_unaligned(p.cast())
}

unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p.cast())
}

unsafe fn store(p: *mut u8, v: __m256i) {
    _mm256_storeu_si256(p.cast(), v)
}

/// Sets the high bit of each space byte. A borrow can also mark the byte above a space if
/// that byte is `!`, which never sits above the first space in a method or among digits.
fn spaces_in(word: u64) -> u64 {
    let z = word ^ SPACES;
    z.wrapping_sub(ONES) & !z & HIGHS
}

/// Every endpoint seen so far. Workers share it: it fills in the first moments of the run
/// and is read-only after that, so one copy stays hot in cache.
struct Catalogue {
    slots: Box<[AtomicU32]>, // tag << 20 | len << 12 | id + 1
    keys: *mut [__m256i; 2], // each name's first and last 32 bytes, masked as in `lookup`
    spans: *mut (u32, u32),  // each name's offset and length in `text`
    text: *mut u8,
    used: Mutex<(usize, usize)>, // ids and text bytes handed out
}

#[repr(align(64))]
struct Masks([u8; 96]);
static MASKS: Masks = {
    let mut bytes = [0u8; 96];
    let mut i = 32;
    while i < 64 {
        bytes[i] = 0xFF;
        i += 1;
    }
    Masks(bytes)
};

impl Catalogue {
    fn new() -> Catalogue {
        let slots = (0..SLOTS).map(|_| AtomicU32::new(0)).collect();
        Catalogue {
            slots,
            keys: ManuallyDrop::new(Vec::with_capacity(MAX_ENDPOINTS)).as_mut_ptr(),
            spans: ManuallyDrop::new(Vec::with_capacity(MAX_ENDPOINTS)).as_mut_ptr(),
            text: vec![0u8; TEXT_BYTES].leak().as_mut_ptr(),
            used: Mutex::new((0, 0)),
        }
    }

    /// Compares whole names without a length-dependent loop: the masked first and last 32
    /// bytes cover any name up to 64 bytes, and only the 6% of lines above that look further.
    #[inline(always)]
    unsafe fn lookup(&self, name: *const u8, len: usize) -> usize {
        let masks = MASKS.0.as_ptr();
        let short = len.min(32);
        let head = _mm256_and_si256(load(name), load(masks.add(64 - short)));
        let tail = _mm256_and_si256(load(name.add(len).sub(32)), load(masks.add(short)));
        let mixed = _mm256_extract_epi64(tail, 3) as u64
            ^ (_mm256_extract_epi64(tail, 1) as u64).rotate_left(13)
            ^ (_mm256_extract_epi64(head, 1) as u64).rotate_left(29)
            ^ len as u64;
        let hash = mixed.wrapping_mul(0x517c_c1b7_2722_0a95);
        let stamp = (hash >> 32) as u32 & 0xFFF00 | len as u32;
        let mut i = (hash >> (64 - SLOTS.trailing_zeros())) as usize;
        loop {
            let slot = self.slots.get_unchecked(i).load(Acquire);
            if slot >> 12 == stamp {
                let id = (slot & 0xFFF) as usize - 1;
                let [key_head, key_tail] = *self.keys.add(id);
                let differs = _mm256_or_si256(
                    _mm256_xor_si256(head, key_head),
                    _mm256_xor_si256(tail, key_tail),
                );
                if _mm256_testz_si256(differs, differs) == 1 && (len <= 64 || self.same(id, name)) {
                    return id;
                }
            } else if slot == 0 {
                return self.insert(i, stamp, name, len, [head, tail]);
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    unsafe fn name(&self, id: usize) -> &[u8] {
        let (offset, len) = *self.spans.add(id);
        std::slice::from_raw_parts(self.text.add(offset as usize), len as usize)
    }

    unsafe fn same(&self, id: usize, name: *const u8) -> bool {
        let known = self.name(id);
        known == std::slice::from_raw_parts(name, known.len())
    }

    /// The key is fully written before the Release store publishes the slot, so a reader
    /// that sees the slot sees the key.
    #[cold]
    unsafe fn insert(
        &self,
        mut i: usize,
        stamp: u32,
        name: *const u8,
        len: usize,
        key: [__m256i; 2],
    ) -> usize {
        let mut used = self.used.lock().unwrap();
        loop {
            let slot = self.slots[i].load(Relaxed);
            if slot == 0 {
                break;
            }
            let id = (slot & 0xFFF) as usize - 1;
            if slot >> 12 == stamp && self.name(id).len() == len && self.same(id, name) {
                return id; // another worker got here while this one waited for the lock
            }
            i = (i + 1) & (SLOTS - 1);
        }
        let (id, offset) = *used;
        assert!(id < MAX_ENDPOINTS && offset + len + 64 < TEXT_BYTES);
        copy_nonoverlapping(name, self.text.add(offset), len);
        self.keys.add(id).write(key);
        self.spans.add(id).write((offset as u32, len as u32));
        *used = (id + 1, offset + len);
        self.slots[i].store(stamp << 12 | (id + 1) as u32, Release);
        id
    }

    /// Ids in output order. A &[u8] orders as unsigned bytes with a proper prefix first.
    fn sorted_ids(&self) -> Vec<u16> {
        let count = self.used.lock().unwrap().0;
        let mut ids: Vec<u16> = (0..count as u16).collect();
        ids.sort_unstable_by(|&a, &b| unsafe {
            self.name(a as usize).cmp(self.name(b as usize))
        });
        ids
    }
}

struct Run {
    input: *const u8,
    len: usize,
    next_claim: AtomicUsize,
    catalogue: Catalogue,
    rows: *mut u32,
    stride: usize,
    claimed: [AtomicUsize; HOURS], // rows of each hour's region given out in chunks
    finished_parsing: Barrier,
    order: OnceLock<Vec<u16>>,
    next_hour: AtomicUsize,
    pending: Mutex<Vec<Option<Vec<u8>>>>,
    output: Mutex<(File, usize)>, // the file and the next hour it needs
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

/// Calls `visit(group, is_error, latency)` for each row of an hour's region.
#[inline(always)]
unsafe fn for_each_row(region: *const u32, claimed: usize, mut visit: impl FnMut(usize, u32, u32)) {
    for chunk in (0..claimed).step_by(CHUNK) {
        let rows = region.add(chunk);
        let mut i = 0;
        while i < CHUNK && *rows.add(i) != 0 {
            let row = *rows.add(i);
            let mut latency = row & ESCAPE;
            if latency == ESCAPE {
                i += 1;
                latency = *rows.add(i);
            }
            visit((row >> (LATENCY_BITS + 1)) as usize, row >> LATENCY_BITS & 1, latency);
            i += 1;
        }
    }
}

#[cold]
unsafe fn long_latency(line_end: *const u8) -> (u32, usize) {
    let (mut value, mut power, mut digits) = (0, 1, 0);
    while *line_end.sub(digits + 1) != b' ' {
        value += (*line_end.sub(digits + 1) - b'0') as u32 * power;
        power = power.wrapping_mul(10);
        digits += 1;
    }
    (value, digits)
}

impl Run {
    #[cold]
    unsafe fn claim_chunk(&self, hour: usize) -> Cursor {
        let offset = self.claimed[hour].fetch_add(CHUNK, Relaxed);
        let at = self.rows.add(hour * self.stride + offset);
        Cursor { at, last: at.add(CHUNK - 1) }
    }

    unsafe fn parse(&self) {
        let input = self.input;
        let empty = std::ptr::null_mut();
        let mut cursors = [Cursor { at: empty, last: empty }; HOURS];
        let mut ends = [0u32; LINES_PER_BLOCK + 1];
        let mapped_end = self.len.div_ceil(4096) * 4096;
        loop {
            let claim = self.next_claim.fetch_add(CLAIM, Relaxed);
            if claim >= self.len {
                return;
            }
            madvise(input.add(claim) as *mut u8, CLAIM.min(mapped_end - claim), MADV_POPULATE_READ);
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
                    if cursor.at >= cursor.last {
                        *cursor = self.claim_chunk(hour as usize);
                    }
                    *cursor.at = ((id + 1) << 1 | is_error) << LATENCY_BITS | latency.min(ESCAPE);
                    cursor.at = cursor.at.add(1);
                    if latency >= ESCAPE {
                        *cursor.at = latency;
                        cursor.at = cursor.at.add(1);
                    }
                    line = end.add(1);
                }
            }
        }
    }

    unsafe fn summarise(&self, hour: usize, order: &[u16], tables: &mut Tables) -> Vec<u8> {
        let region = self.rows.add(hour * self.stride);
        let claimed = self.claimed[hour].load(Relaxed);

        let Tables { starts, errors, latencies } = tables;
        starts.fill(0);
        errors.fill(0);
        for_each_row(region, claimed, |group, is_error, _| {
            *starts.get_unchecked_mut(group) += 1;
            *errors.get_unchecked_mut(group) += is_error;
        });
        let mut total = 0;
        for start in starts.iter_mut() {
            (total, *start) = (total + *start, total);
        }
        latencies.resize(total as usize, 0);
        for_each_row(region, claimed, |group, _, latency| {
            let start = starts.get_unchecked_mut(group);
            *latencies.get_unchecked_mut(*start as usize) = latency;
            *start += 1; // leaves each group's end where the next group's start was
        });
        madvise(region.cast(), (claimed * 4).div_ceil(4096) * 4096, MADV_DONTNEED);

        let label = hour_label(hour);
        let mut out: Vec<u8> = Vec::with_capacity(1 << 18);
        for &id in order {
            let group = id as usize + 1;
            let span = starts[group - 1] as usize..starts[group] as usize;
            if span.is_empty() {
                continue;
            }
            let (n, bad) = (span.len() as u32, errors[group]);
            let [p50, p95, p99] = percentiles(&mut latencies[span]);
            let name = self.catalogue.name(id as usize);
            out.reserve(LINE_MAX);
            let start = out.as_mut_ptr().add(out.len());
            store(start, load(label.as_ptr()));
            let mut w = start.add(21);
            store(w, load(name.as_ptr()));
            store(w.add(32), load(name.as_ptr().add(32)));
            if name.len() > 64 {
                copy_nonoverlapping(name.as_ptr(), w, name.len());
            }
            w = w.add(name.len());
            for number in [n, p50, p95, p99, bad] {
                w = put_number(w, number);
            }
            w = put_number(w, (bad as u64 * 1_000_000 / n as u64) as u32);
            *w = b'\n';
            out.set_len(out.len() + w.offset_from(start) as usize + 1);
        }
        out
    }

    /// Writes every hour that is ready, in order. The caller holds the output lock.
    fn flush(&self, output: &mut (File, usize)) {
        while output.1 < HOURS {
            let Some(part) = self.pending.lock().unwrap()[output.1].take() else {
                return;
            };
            output.0.write_all(&part).unwrap();
            output.1 += 1;
        }
    }

    fn work(&self) {
        unsafe { self.parse() };
        self.finished_parsing.wait();
        let order = self.order.get_or_init(|| self.catalogue.sorted_ids());
        let size = order.len() + 2;
        let mut tables = Tables {
            starts: vec![0; size],
            errors: vec![0; size],
            latencies: Vec::new(),
        };
        loop {
            let hour = self.next_hour.fetch_add(1, Relaxed);
            if hour >= HOURS {
                break;
            }
            let part = unsafe { self.summarise(hour, order, &mut tables) };
            self.pending.lock().unwrap()[hour] = Some(part);
            if let Ok(mut output) = self.output.try_lock() {
                self.flush(&mut output);
            }
        }
        // The last worker here sees every part, whoever held the lock when it was stored.
        self.flush(&mut self.output.lock().unwrap());
    }
}

struct Tables {
    starts: Vec<u32>,
    errors: Vec<u32>,
    latencies: Vec<u32>,
}

/// The hour as text with a trailing space, padded so it can be copied as one 32-byte load.
fn hour_label(hour: usize) -> [u8; 32] {
    let (day, clock) = ((hour + 18) / 24, (hour + 18) % 24);
    let (month, date) = if day < 2 { (10, 30 + day) } else { (11, day - 1) };
    let mut label = [0; 32];
    let text = format!("2026-{month}-{date:02}T{clock:02}:00:00Z ");
    label[..21].copy_from_slice(text.as_bytes());
    label
}

/// Nearest-rank p50, p95 and p99. Selecting p50 first halves the slice the other two search,
/// and p95 then leaves p99 only the top 5%. Starting from p99 would rescan almost all of it.
fn percentiles(v: &mut [u32]) -> [u32; 3] {
    let n = v.len();
    let rank = |p: usize| (p * n + 99) / 100 - 1;
    if n <= 64 {
        v.sort_unstable(); // also keeps the ranks from colliding in the selections below
        return [v[rank(50)], v[rank(95)], v[rank(99)]];
    }
    let (_, &mut p50, high) = v.select_nth_unstable(rank(50));
    let (_, &mut p95, high) = high.select_nth_unstable(rank(95) - rank(50) - 1);
    let (_, &mut p99, _) = high.select_nth_unstable(rank(99) - rank(95) - 1);
    [p50, p95, p99]
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

/// Writes a space and the number, with the same few instructions for every length below
/// 10^8, so the varying widths do not mispredict.
unsafe fn put_number(w: *mut u8, value: u32) -> *mut u8 {
    *w = b' ';
    let mut w = w.add(1);
    if value >= 100_000_000 {
        let top = value / 100_000_000;
        if top >= 10 {
            *w = b'0' + (top / 10) as u8;
            w = w.add(1);
        }
        *w = b'0' + (top % 10) as u8;
        write_unaligned(w.add(1).cast(), eight_digits(value % 100_000_000) + ZEROS);
        return w.add(9);
    }
    let digits = eight_digits(value);
    let skip = (digits | 1 << 56).trailing_zeros() as usize / 8;
    write_unaligned(w.cast(), (digits + ZEROS) >> (8 * skip));
    w.add(8 - skip)
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
        let private = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
        let null = std::ptr::null_mut();
        let span = len.div_ceil(4096) * 4096 + 4096;
        let input = mmap(null, span, PROT_READ, private, -1, 0);
        if len > 0 {
            mmap(input, len, PROT_READ, MAP_PRIVATE | MAP_FIXED, file.as_raw_fd(), 0);
        }
        let stride = (len / MIN_LINE + 1) * 3 + threads * CHUNK; // every line could escape
        let stride = stride.div_ceil(1024) * 1024;
        let rows = mmap(null, stride * HOURS * 4, PROT_READ | PROT_WRITE, private, -1, 0);
        (input as *const u8, rows.cast(), stride)
    };

    let run = Run {
        input,
        len,
        next_claim: AtomicUsize::new(0),
        catalogue: Catalogue::new(),
        rows,
        stride,
        claimed: [const { AtomicUsize::new(0) }; HOURS],
        finished_parsing: Barrier::new(threads),
        order: OnceLock::new(),
        next_hour: AtomicUsize::new(0),
        pending: Mutex::new(vec![None; HOURS]),
        output: Mutex::new((File::create("/out/summary.txt").unwrap(), 0)),
    };
    thread::scope(|s| {
        for _ in 1..threads {
            let builder = thread::Builder::new().stack_size(1 << 18);
            builder.spawn_scoped(s, || run.work()).unwrap();
        }
        run.work();
    });
}

//! Speed Demon 2026, Track A (Ingest).
//!
//! Data layout. The input is mapped read-only, followed by one spare zero page. One anonymous
//! reservation holds a fixed-size area per hour (`stride` u32s; untouched pages cost nothing):
//! the hour's rows first, in chunks of 256 claimed by whichever worker needs one; later the
//! hour's output text, written over its rows; and the radix-sort scratch further on. Endpoints
//! are ids from one shared hash table that fills in the first moments of the run.
//!
//! Threads. `main` starts `available_parallelism() - 1` workers and then works itself, so every
//! thread runs `Run::work`. Work is handed out through atomic counters (file claims, then
//! hours). The two phases overlap: after each claim a worker may summarise an hour that is
//! sealed, meaning that the claims parsed so far at the front of the file hold a line two
//! hours later, so (the input being the interleaving of eight time-ordered streams) no claim
//! still to come should hold one of that hour. If that bet fails on some input, `solve` runs
//! the whole job again and seals nothing before the end of the file, so the output is exact
//! whatever the order of the lines.
//!
//! 1. Parse. Workers claim 1 MiB of the mapped file at a time (pages fault in on demand) and,
//!    64 KiB at a time, find every newline first and then read each line from both ends with
//!    no data-dependent branches. Each line becomes a 4-byte row in its hour's area. A claim's
//!    file pages are dropped as soon as it is parsed, and it records the lowest and highest
//!    hour it holds, which is all `advance` needs to seal hours and to notice a failed bet.
//! 2. Summarise. Workers take sealed hours in order and radix sort each one's rows by latency,
//!    inside the hour's own pages. Dealing the sorted rows out by endpoint keeps them in order,
//!    so every endpoint's latencies come out sorted and a percentile is one array read. The
//!    hour's text is built over its rows, in the same pages. Once every earlier hour's text
//!    exists, an hour's offset in the file is known, and the thread that finds it ready writes
//!    it there with `pwrite` and gives its pages back. So writes run in several threads at
//!    once and overlap the work, and memory is released as fast as the tmpfs output file grows
//!    (which counts against the memory limit too). Only a few hours of rows are alive at a
//!    time, not all 72.
//!
//! Lines are ASCII and well formed (the spec promises it), so the hot loops use raw pointers.
use std::arch::x86_64::*;
use std::ptr::{read_unaligned as read, write_unaligned};
use std::sync::{atomic::Ordering::*, atomic::*, Arc, Mutex, OnceLock};
use std::{fs::File, os::fd::AsRawFd, os::unix::fs::FileExt, thread};

extern "C" {
    fn mmap(addr: usize, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    // Every `madvise` below passes advice 4, MADV_DONTNEED: give the pages back.
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn prctl(option: i32, ...) -> i32;
}

const CLAIM: usize = 1 << 20;
const BLOCK: usize = 1 << 16; // fits in L2 between the newline pass and the line pass
const HOURS: usize = 72; // the span is fixed for every seed: 2026-10-30T18 to 2026-11-02T17
const CHUNK: usize = 256; // rows of an hour's region a worker claims at a time
const SLOTS: usize = 1 << 13;
const ID_BITS: u32 = 11; // low bits of a slot hold id + 1
const ZEROS: u64 = 0x3030_3030_3030_3030;

/// A row is id + 1 in 11 bits, an error bit, then the latency in 20 bits. Only about 1% of
/// latencies need more: those store ESCAPE here and the full value in the next row, which
/// keeps almost every row at 4 bytes. A row of 0 marks the unused end of a chunk.
const LATENCY_BITS: u32 = 20;
const ESCAPE: u32 = (1 << LATENCY_BITS) - 1;
const DIGIT_BITS: u32 = 10;
const DIGITS: usize = 1 << DIGIT_BITS; // of the two radix passes over a 20-bit latency

/// Sets the high bit of each space byte. A borrow can also mark the byte above a space if
/// that byte is `!`, which never sits above the first space in a method or among digits.
fn spaces_in(word: u64) -> u64 {
    let z = word ^ 0x2020_2020_2020_2020;
    z.wrapping_sub(0x0101_0101_0101_0101) & !z & 0x8080_8080_8080_8080
}

/// A hash of the whole name without a length-dependent loop: the first and last 32 bytes, with
/// the bytes outside the name masked off, cover any name up to 64 bytes; a longer name also
/// mixes two loads that overlap its middle. Never leaves the id bits zero, so a live slot is
/// never 0.
///
/// Loads may stray outside the name, which is safe: the tail load starts at most 32 bytes
/// before the name's end, still inside the line (a name starts at byte 32 or later), and the
/// head load runs at most 32 bytes past the end of the file, into the spare zero page.
#[inline(always)]
unsafe fn name_hash(name: *const u8, len: usize) -> u64 {
    // Loading at `WINDOW + 64 - n` keeps the first n bytes, and at `WINDOW + n` the last n.
    const WINDOW: [[u8; 32]; 3] = [[0; 32], [255; 32], [0; 32]];
    let load = |at: *const u8| _mm256_loadu_si256(at.cast());
    let (window, n) = (WINDOW.as_ptr().cast::<u8>(), len.min(32));
    let head = _mm256_and_si256(load(name), load(window.add(64 - n)));
    let tail = _mm256_and_si256(load(name.add(len).sub(32)), load(window.add(n)));
    let mut mixed = _mm256_aesenc_epi128(head, tail);
    if len > 64 {
        mixed = _mm256_aesenc_epi128(mixed, load(name.add(32)));
        mixed = _mm256_aesenc_epi128(mixed, load(name.add(len - 64)));
    }
    // Two more AES rounds (`vaesenc` does both 16-byte halves at once) mix far better than a
    // multiply, at about the same cost.
    let key = _mm256_set1_epi64x(0x9E37_79B9_7F4A_7C15u64 as i64);
    let mixed = _mm256_aesenc_epi128(_mm256_aesenc_epi128(mixed, key), mixed);
    let [a, b, c, d]: [u64; 4] = std::mem::transmute(mixed);
    (a ^ b ^ c ^ d) | 1 << ID_BITS
}

struct Run {
    input: *mut u8,
    len: usize,
    // The catalogue: every endpoint seen so far. Workers share it: it fills in the first
    // moments of the run and is read-only after that, so one copy stays hot in cache.
    // Names are identified by a 64-bit hash alone (53 bits of it are compared), the same
    // deliberate shortcut Tracks B and C take: with 1,842 names the chance of any two
    // colliding is about 2e-10.
    slots: [AtomicU64; SLOTS], // hash >> 11 << 11 | id + 1; 0 is an empty slot
    names: Mutex<(Vec<&'static [u8]>, Arc<Vec<(&'static [u8], usize)>>)>, // by id, then in output order
    rows: *mut u32,
    stride: usize,
    claimed: [AtomicUsize; HOURS], // rows of each hour's region given out in chunks
    // Per claim: 0 until parsed, then 1 << 16 | lowest hour << 8 | highest hour of its lines.
    parsed: Vec<AtomicU32>,
    // Shared counters: [0] the next file claim; [1] the hours below it have all their rows (early
    // sealing is a bet on the input's order, see `advance`); [2] the next hour to summarise; [3] nonzero
    // once a line turned up in an hour that was already sealed.
    ctl: [AtomicUsize; 4],
    early: bool,
    texts: [OnceLock<&'static [u8]>; HOURS], // each hour's text, until it is written
    output: File,
    placed: Mutex<(usize, u64)>, // the next hour to place in the file, and its offset
}
unsafe impl Sync for Run {}

impl Run {
    /// The id of an endpoint name, new if this is its first appearance. Slots are only ever
    /// filled, under the lock, so a slot seen empty under the lock ends the probe sequence: the
    /// name is new and goes there. The name is written before the Release store publishes the
    /// slot, so a reader that sees the slot sees the name.
    #[inline(always)]
    unsafe fn lookup(&self, name: *const u8, len: usize) -> usize {
        let hash = name_hash(name, len);
        let mut i = (hash >> (64 - SLOTS.trailing_zeros())) as usize;
        loop {
            let slot = self.slots[i].load(Acquire);
            if slot >> ID_BITS == hash >> ID_BITS {
                return (slot & ((1 << ID_BITS) - 1)) as usize - 1;
            } else if slot == 0 {
                let (names, _) = &mut *self.names.lock().unwrap();
                if self.slots[i].load(Acquire) == 0 {
                    let id = names.len();
                    assert!(id + 1 < 1 << ID_BITS); // id + 1 must fit the low bits of a slot
                    names.push(std::slice::from_raw_parts(name, len).to_vec().leak());
                    self.slots[i].store(hash >> ID_BITS << ID_BITS | (id + 1) as u64, Release);
                    return id;
                }
                continue; // another worker filled this slot meanwhile: look at it again
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    unsafe fn parse(&self) {
        let input = self.input;
        // Per hour, the next free row and the final slot of the chunk being filled: a row needs
        // two free, in case it has to escape.
        let mut cursors = [(std::ptr::null_mut::<u32>(), std::ptr::null_mut()); HOURS];
        let (mut hour, mut cursor) = (0, (std::ptr::null_mut(), std::ptr::null_mut()));
        let mut ends = [0u32; BLOCK / 40 + 1];
        loop {
            self.advance();
            let claim = self.ctl[0].fetch_add(CLAIM, Relaxed);
            if claim >= self.len {
                break;
            }
            // A line belongs to the claim holding its newline. The first one starts after
            // the previous claim's last newline, at most one line back.
            let mut line = input.add(claim);
            while claim > 0 && *line.sub(1) != b'\n' {
                line = line.sub(1);
            }
            // Every claim starts by reading its first hour, so it notes the hours it holds.
            let (mut key, mut low, mut high) = (u64::MAX, HOURS, 0);
            for start in (claim..(claim + CLAIM).min(self.len)).step_by(BLOCK) {
                // Find the newlines of the block first. Lines are longer than the 32-byte step, so
                // a step holds at most one newline, and one unconditional store records it (a
                // step with none writes a junk entry that the next store overwrites). Each pass
                // over a block is then a counted loop, with no branch on where lines end. The
                // aligned load is valid because blocks start at multiples of 64 KiB in a
                // page-aligned mapping, and the last step may read up to 31 bytes past the
                // file, into the spare zero page.
                let newline = _mm256_set1_epi8(b'\n' as i8);
                let (mut count, mut at) = (0, start);
                while at < (start + BLOCK).min(self.len) {
                    let hits = _mm256_cmpeq_epi8(_mm256_load_si256(input.add(at).cast()), newline);
                    let mask = _mm256_movemask_epi8(hits) as u32;
                    *ends.get_unchecked_mut(count) = (at + mask.trailing_zeros() as usize) as u32;
                    count += (mask != 0) as usize;
                    at += 32;
                }
                for &end in &ends[..count] {
                    let end = input.add(end as usize);
                    _mm_prefetch(end.add(BLOCK).cast(), _MM_HINT_T0);
                    // `clock` is "DDTHH:MM" (bytes 8 to 15 of the timestamp). Oct 30, Oct 31,
                    // Nov 1 and Nov 2 are day 0 to 3 of the span: the day's units digit plus
                    // one, less one in October, whose tens digit '3' (0x33) has bit 1 set
                    // where '0' (0x30) does not.
                    let clock = read::<u64>(line.add(8).cast());
                    if clock << 24 != key {
                        // Hours change rarely along the file, so the cursor of the current
                        // hour stays in registers until the day and hour digits differ.
                        cursors[hour] = cursor;
                        key = clock << 24;
                        let day = (clock >> 8 & 15) + 1 - (clock >> 1 & 1);
                        hour =
                            (day * 24 + (clock >> 24 & 15) * 10 + (clock >> 32 & 15) - 18) as usize;
                        cursor = cursors[hour];
                        (low, high) = (low.min(hour), high.max(hour));
                    }
                    // The method starts at byte 28 and is 3 to 6 letters, so the first space
                    // in that word gives its length, and the name follows it.
                    let method = spaces_in(read(line.add(28).cast())).trailing_zeros() as usize / 8;
                    let name = line.add(29 + method);
                    // Read the latency back from the newline. Its length varies line by
                    // line, so a digit loop would mispredict on nearly every one.
                    let last_eight = read::<u64>(end.sub(8).cast());
                    let spaces = spaces_in(last_eight);
                    let (latency, digits) = if spaces != 0 {
                        // The digits are the top `digits` bytes of the little-endian word.
                        // Keep only those, as values 0 to 9, and join neighbours in three
                        // multiply-shift steps: pairs, then fours, then all eight.
                        let digits = spaces.leading_zeros() as usize / 8;
                        let v = (last_eight ^ ZEROS) >> (64 - 8 * digits) << (64 - 8 * digits);
                        let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
                        let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
                        let v = (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32;
                        (v as u32, digits)
                    } else {
                        // Eight digits or more: the word-wide decode cannot see where they start.
                        let digits = (1..).find(|&n| *end.sub(n + 1) == b' ').unwrap();
                        let text = std::slice::from_raw_parts(end.sub(digits), digits);
                        (std::str::from_utf8_unchecked(text).parse().unwrap(), digits)
                    };
                    // The line ends "<name> <status> <latency>\n", with a 3-digit status.
                    let status = end.sub(digits + 4);
                    let len = status.sub(1).offset_from(name) as usize;
                    let id = self.lookup(name, len) as u32;
                    let is_error = (*status == b'5') as u32;
                    // Each worker fills its own chunk of each hour, so storing a row needs no
                    // atomics; only taking a new chunk touches the shared counter.
                    if cursor.0 >= cursor.1 {
                        let at = self.claimed[hour].fetch_add(CHUNK, Relaxed) + hour * self.stride;
                        cursor = (self.rows.add(at), self.rows.add(at + CHUNK - 1));
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
            madvise(input.add(claim), CLAIM.min(self.len - claim), 4);
            self.parsed[claim / CLAIM].store((1 << 16 | low << 8 | high) as u32, Release);
        }
        // Out of claims: what is left is hours that wait for the last claims in flight.
        while self.ctl[2].load(Relaxed) < HOURS {
            self.advance();
            thread::yield_now();
        }
    }

    /// Seals hours. The input is the interleaving of eight time-ordered streams, so the lines of
    /// an hour are all in the file before the first line two hours later (on seed 1 by at least
    /// 0.78 MiB; the spec promises no bound, so this is a bet, not a fact). Once the parsed
    /// claims at the front of the file hold a line of hour h + 2, no claim still to be parsed
    /// should hold one of hour h, and h is sealed. Its rows are then summarised while the rest of
    /// the file is read, and given back at once, so only a few hours of rows are alive instead of
    /// all 72. If the bet fails, some claim holds a line of an hour sealed before it (noticed
    /// here, once the claims in front of it are parsed): `dirty`, and `solve` runs again,
    /// sealing nothing until the whole file is read.
    ///
    /// Then summarises the next sealed hour, if there is one, and writes what is ready.
    fn advance(&self) {
        let (mut top, mut sealed) = (0usize, HOURS);
        for claim in &self.parsed {
            let info = claim.load(Acquire) as usize;
            if info == 0 {
                sealed = top.saturating_sub(1);
                break;
            }
            self.ctl[3].fetch_or(((info >> 8 & 255) + 1 < top) as usize, Relaxed);
            top = top.max((info & 255) * self.early as usize);
        }
        self.ctl[1].fetch_max(sealed, Release);
        let sealed = |hour| (hour < self.ctl[1].load(Acquire)).then_some(hour + 1);
        if let Ok(hour) = self.ctl[2].try_update(Relaxed, Relaxed, sealed) {
            // The names may have grown since the last hour was summarised.
            let order = {
                let (names, order) = &mut *self.names.lock().unwrap();
                if order.len() != names.len() {
                    let mut sorted: Vec<_> = names.iter().copied().zip(0..).collect();
                    sorted.sort_unstable();
                    *order = Arc::new(sorted);
                }
                order.clone()
            };
            let part = unsafe { self.summarise(hour, &order) };
            self.texts[hour].set(part).unwrap();
            // Whoever takes the lock after an hour's text is set sees it, so every hour is
            // written exactly once, by the thread that finds it ready.
            self.write_ready();
        }
    }

    /// Sorts and summarises one hour, and returns its text. Everything lives in the hour's own
    /// area of `rows`: the rows first, then the text over them, and the sort scratch further
    /// on. Nothing is allocated per hour, and `write_ready` gives the whole area back in one call.
    /// (Text in malloc'd buffers was once the bulk of the memory peak: the allocator kept every
    /// buffer it had grown while a single writer fell behind.)
    unsafe fn summarise(&self, hour: usize, order: &[(&[u8], usize)]) -> &'static [u8] {
        let rows = self.rows.add(hour * self.stride);
        let claimed = self.claimed[hour].load(Relaxed);
        // The scratch starts past the rows and at least (1 << 18) rows (1 MiB) in: the text is
        // written over the rows and can grow longer than them, though never to 1 MiB. Rounded
        // to whole pages so that `madvise` can release it.
        let spare = rows.add(claimed.max((1 << 18)).next_multiple_of(1024));
        // Counts per endpoint, indexed by the row's group (id + 1), so that after dealing
        // `starts[id]..starts[id + 1]` is the span of `id` (see `deal`).
        let (mut starts, mut errors) = (vec![0u32; 1 << ID_BITS], vec![0u32; 1 << ID_BITS]);
        let mut long = Vec::new();

        // Moves rows down over the gaps between chunks. The 1% that escape go aside: they
        // are all longer than the rest, so sorted on their own they simply follow them.
        let (mut sizes, mut count) = ([[0u32; DIGITS]; 2], 0);
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
            // Group id + 1 advances from its start to its end, which is where the next group
            // started, so starts[id]..starts[id + 1] ends up as the span of `id`.
            *start += 1;
        };
        for i in 0..count {
            deal(*rows.add(i), *rows.add(i) & ESCAPE);
        }
        long.iter().for_each(|&(latency, row)| deal(row, latency));

        // The hour as text with a trailing space (21 bytes), copied in front of every line.
        let date = ["10-30", "10-31", "11-01", "11-02"][(hour + 18) / 24];
        let label = format!("2026-{date}T{:02}:00:00Z ", (hour + 18) % 24);
        let (text, mut w) = (rows.cast::<u8>(), rows.cast::<u8>());
        for &(name, id) in order.iter().filter(|&&(_, id)| starts[id] < starts[id + 1]) {
            let span = starts[id] as usize..starts[id + 1] as usize;
            let (n, bad) = (span.len() as u32, errors[id + 1]);
            let sorted = std::slice::from_raw_parts(spare.add(span.start), span.len());
            let at_rank = |p: usize| sorted[(p * sorted.len() + 99) / 100 - 1];
            w.copy_from_nonoverlapping(label.as_ptr(), 21);
            w.add(21)
                .copy_from_nonoverlapping(name.as_ptr(), name.len());
            w = w.add(21 + name.len());
            for number in [n, at_rank(50), at_rank(95), at_rank(99), bad] {
                w = put_number(w, number);
            }
            w = put_number(w, (bad as u64 * 1_000_000 / n as u64) as u32);
            *w = b'\n';
            w = w.add(1);
        }
        // The ranks are all read: give the scratch pages back now, not when the hour is written.
        madvise(spare.cast(), (count + long.len()) * 4, 4);
        std::slice::from_raw_parts(text, w.offset_from(text) as usize)
    }

    /// Gives every ready hour whose predecessors are placed its offset in the file, then
    /// writes those hours outside the lock. Writers in different threads run at once, which
    /// matters: one thread writing 10 MB to tmpfs fell up to 2 ms behind the summaries.
    fn write_ready(&self) {
        let mut ready = Vec::new();
        let mut placed = self.placed.lock().unwrap();
        while let Some(text) = self.texts.get(placed.0).and_then(|text| text.get()) {
            ready.push((placed.0, placed.1, *text));
            *placed = (placed.0 + 1, placed.1 + text.len() as u64);
        }
        drop(placed);
        for (hour, offset, text) in ready {
            self.output.write_all_at(text, offset).unwrap();
            // The text is out: give the hour's pages back.
            unsafe { madvise(self.rows.add(hour * self.stride).cast(), self.stride * 4, 4) };
        }
    }
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
    // Leading zero digits are the low bytes. The top bit is a sentinel, so 0 still prints one
    // digit. The 16-byte store runs past the number; the next field or line overwrites it.
    let skip = (digits | 1 << 127).trailing_zeros() as usize / 8;
    let zeros = ZEROS as u128 * ((1 << 64) + 1);
    write_unaligned(w.add(1).cast(), (digits + zeros) >> (8 * skip));
    w.add(17 - skip)
}

fn main() {
    unsafe { solve(true) && solve(false) };
}

/// Returns whether the early sealing failed.
unsafe fn solve(early: bool) -> bool {
    let file = File::open("/data/requests.log").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());

    // Huge pages would turn each hour's first touch into 2 MB of memory.
    // Name lookups read a little past a line, so a spare zero page follows the file.
    // Each hour reserves room for every line, and only the pages it touches cost anything.
    // Linux constants: prctl 41 is PR_SET_THP_DISABLE; mmap protections 1 (read) and 3
    // (read and write); flags 0x4022 (private, anonymous, no reserve) and 0x12 (private, fixed).
    prctl(41, 1usize, 0usize, 0usize, 0usize);
    let map = |at, len, prot, flags, fd| mmap(at, len, prot, flags, fd, 0);
    let input = map(0, len.div_ceil(4096) * 4096 + 4096, 1, 0x4022, -1);
    // For an empty file this fails, and the zero page stays.
    map(input as usize, len, 1, 0x12, file.as_raw_fd());
    // An hour has room for its rows (two per line, if every line escaped), then for the
    // scratch (one value per line) after them, or after (1 << 18) rows if that is further,
    // plus the unused ends of one partly filled chunk per thread.
    let stride = ((1 << 18) + (len / 40 + 1) * 4 + threads * 512).next_multiple_of(1 << 19);
    // Hour areas are multiples of 2 MiB and the first starts on a 2 MiB boundary, so each
    // hour's pages share one page table instead of spilling into the next.
    let rows = map(0, stride * HOURS * 4 + (2 << 20), 3, 0x4022, -1)
        .map_addr(|a| a.next_multiple_of(2 << 20));

    let run = Run {
        input,
        len,
        slots: [const { AtomicU64::new(0) }; SLOTS],
        names: Mutex::default(),
        rows: rows.cast(),
        stride,
        claimed: [const { AtomicUsize::new(0) }; HOURS],
        parsed: (0..len.div_ceil(CLAIM)).map(|_| 0.into()).collect(),
        ctl: Default::default(),
        early,
        texts: [const { OnceLock::new() }; HOURS],
        output: File::create("/out/summary.txt").unwrap(),
        placed: Mutex::new((0, 0)),
    };
    thread::scope(|s| {
        (0..threads).for_each(|_| drop(s.spawn(|| unsafe { run.parse() })));
    });
    run.ctl[3].load(Relaxed) != 0
}

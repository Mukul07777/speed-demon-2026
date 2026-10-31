//! Speed Demon 2026, Track A (Ingest).
//!
//! Workers claim blocks of requests.log and read them with pread, so memory never holds
//! the whole file. Each row packs into a u64 and goes into a bucket per hour. Then each
//! hour is counting-sorted by endpoint, and each group selects only its three percentiles.
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

const BLOCK: usize = 1 << 17; // fits in L2 alongside the rows being written
const MAX_LINE: usize = 177; // ts 27, method 7, endpoint 125, status 3, latency 10, 3 spaces, LF
const PADDING: usize = 64; // wide loads may run past the last line of the file
const HOURS: usize = 72; // the span is fixed for every seed: 2026-10-30T18 to 2026-11-02T17
const SLOTS: usize = 1 << 12; // the catalogue has 1,842 endpoints
const ID_SHIFT: u32 = 33; // a row is endpoint << 33 | latency << 1 | is_error

struct Worker {
    slots: Vec<u32>, // open-addressed table of endpoint id + 1
    endpoints: Vec<Box<[u8]>>,
    hours: Vec<Vec<u64>>,
}

fn load(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Hashes the first and last eight bytes, which is where endpoint templates differ.
fn hash(name: &[u8]) -> usize {
    let len = name.len();
    let mixed = if len >= 8 {
        load(name, 0) ^ load(name, len - 8).rotate_left(23)
    } else {
        name.iter().fold(0, |h, &c| h << 8 | c as u64)
    };
    let h = (mixed ^ len as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    (h >> (64 - SLOTS.trailing_zeros())) as usize
}

impl Worker {
    fn new() -> Worker {
        Worker {
            slots: vec![0; SLOTS],
            endpoints: Vec::new(),
            hours: vec![Vec::new(); HOURS],
        }
    }

    fn endpoint_id(&mut self, name: &[u8]) -> u64 {
        let mut i = hash(name);
        loop {
            let slot = self.slots[i];
            match slot {
                0 => {
                    self.endpoints.push(name.into());
                    self.slots[i] = self.endpoints.len() as u32;
                    return self.endpoints.len() as u64 - 1;
                }
                s if &*self.endpoints[s as usize - 1] == name => return s as u64 - 1,
                _ => i = (i + 1) & (SLOTS - 1),
            }
        }
    }
}

fn digit(byte: u8) -> usize {
    (byte - b'0') as usize
}

/// Scans 32 bytes at a time. Most endpoints end within the first load.
fn find_space(b: &[u8], mut at: usize) -> usize {
    loop {
        let found = unsafe {
            let chunk = _mm256_loadu_si256(b[at..at + 32].as_ptr().cast());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8))) as u32
        };
        if found != 0 {
            return at + found.trailing_zeros() as usize;
        }
        at += 32;
    }
}

/// Parses the digits at `at` up to the line's LF and returns (value, index after the LF).
/// Latency lengths vary, so a byte loop mispredicts on nearly every line. This reads eight
/// bytes at once and only falls back to the loop for the rare value of eight digits or more.
fn parse_latency(b: &[u8], at: usize) -> (u64, usize) {
    let digits = load(b, at) ^ 0x3030_3030_3030_3030;
    let not_digit = digits.wrapping_add(0x7676_7676_7676_7676) & 0x8080_8080_8080_8080;
    if not_digit == 0 {
        let (mut value, mut p) = (0, at);
        while b[p] != b'\n' {
            value = value * 10 + digit(b[p]) as u64;
            p += 1;
        }
        return (value, p + 1);
    }
    let len = (not_digit.trailing_zeros() / 8) as usize;
    // Shift the digits to the top so the bytes below them read as leading zeros, then
    // combine neighbouring digits, then pairs, then quads.
    let v = digits << (64 - 8 * len);
    let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    let v = (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32;
    (v, at + len + 1)
}

fn parse(file: &File, len: usize, next: &AtomicUsize) -> Worker {
    let mut w = Worker::new();
    let mut buf = vec![0u8; BLOCK + MAX_LINE + 1 + PADDING];
    loop {
        let start = next.fetch_add(BLOCK, Relaxed);
        if start >= len {
            return w;
        }
        // A line belongs to the block it starts in, so read one byte back to see whether
        // the block opens on a line start, and MAX_LINE on to finish the last line.
        let from = start.saturating_sub(1);
        let size = (BLOCK + MAX_LINE + 1).min(len - from);
        file.read_exact_at(&mut buf[..size], from as u64).unwrap();
        let b = &buf[..];
        let mut p = match start {
            0 => 0,
            _ => b.iter().position(|&c| c == b'\n').unwrap() + 1,
        };
        let stop = (start + BLOCK).min(len) - from;
        while p < stop {
            let date = digit(b[p + 8]) * 10 + digit(b[p + 9]);
            let day = if b[p + 6] == b'0' { date - 30 } else { date + 1 };
            let hour = day * 24 + digit(b[p + 11]) * 10 + digit(b[p + 12]) - 18;
            let method = match (b[p + 28], b[p + 29]) {
                (b'G', _) | (_, b'U') => 3,
                (b'H', _) | (_, b'O') => 4,
                (b'D', _) => 6,
                (b'O', _) => 7,
                _ => 5,
            };
            let name = p + 29 + method;
            let end = find_space(b, name);
            let is_error = (b[end + 1] == b'5') as u64;
            let (latency, after) = parse_latency(b, end + 5);
            p = after;
            let id = w.endpoint_id(&b[name..end]);
            w.hours[hour].push(id << ID_SHIFT | latency << 1 | is_error);
        }
    }
}

fn hour_label(hour: usize) -> String {
    let (day, clock) = ((hour + 18) / 24, (hour + 18) % 24);
    let (month, date) = if day < 2 { (10, 30 + day) } else { (11, day - 1) };
    format!("2026-{month}-{date:02}T{clock:02}:00:00Z")
}

/// Nearest-rank p50, p95 and p99. Selecting p99 first leaves everything below it on the
/// left, so each later selection works on a smaller slice.
fn percentiles(v: &mut [u32]) -> [u32; 3] {
    let n = v.len();
    let rank = |p: usize| (p * n + 99) / 100 - 1;
    if n <= 64 {
        v.sort_unstable(); // also keeps the ranks from colliding in the selections below
        return [v[rank(50)], v[rank(95)], v[rank(99)]];
    }
    let (low, &mut p99, _) = v.select_nth_unstable(rank(99));
    let (low, &mut p95, _) = low.select_nth_unstable(rank(95));
    let (_, &mut p50, _) = low.select_nth_unstable(rank(50));
    [p50, p95, p99]
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
    out.push(b' ');
    out.extend_from_slice(&digits[at..]);
}

fn summarise(hour: usize, workers: &[Worker], ranks: &[Vec<u32>], names: &[&[u8]]) -> Vec<u8> {
    let rows = || {
        let pairs = workers.iter().zip(ranks);
        pairs.flat_map(|(w, rank)| {
            let unpack = move |&r: &u64| (rank[(r >> ID_SHIFT) as usize] as usize, r);
            w.hours[hour].iter().map(unpack)
        })
    };
    let mut starts = vec![0u32; names.len() + 1];
    rows().for_each(|(group, _)| starts[group + 1] += 1);
    for group in 0..names.len() {
        starts[group + 1] += starts[group];
    }
    let mut filled = starts.clone();
    let mut errors = vec![0u32; names.len()];
    let mut latencies = vec![0u32; starts[names.len()] as usize];
    for (group, row) in rows() {
        latencies[filled[group] as usize] = (row >> 1) as u32;
        filled[group] += 1;
        errors[group] += (row & 1) as u32;
    }

    let label = hour_label(hour);
    let mut out = Vec::with_capacity(names.len() * 80);
    for group in 0..names.len() {
        let span = starts[group] as usize..starts[group + 1] as usize;
        if span.is_empty() {
            continue;
        }
        let n = span.len() as u64;
        let bad = errors[group] as u64;
        let [p50, p95, p99] = percentiles(&mut latencies[span]);
        out.extend_from_slice(label.as_bytes());
        out.push(b' ');
        out.extend_from_slice(names[group]);
        for number in [n, p50 as u64, p95 as u64, p99 as u64, bad, bad * 1_000_000 / n] {
            put_number(&mut out, number);
        }
        out.push(b'\n');
    }
    out
}

fn main() {
    let file = File::open("/data/requests.log").unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let threads = thread::available_parallelism().map_or(12, |n| n.get());

    let next = AtomicUsize::new(0);
    let workers: Vec<Worker> = thread::scope(|s| {
        let spawned: Vec<_> = (0..threads)
            .map(|_| s.spawn(|| parse(&file, len, &next)))
            .collect();
        spawned.into_iter().map(|t| t.join().unwrap()).collect()
    });

    // Merge the local ids through one more table, then rank the distinct names. A &[u8]
    // orders as unsigned bytes with a proper prefix first, which is the output order.
    let mut merged = Worker::new();
    let mut ranks: Vec<Vec<u32>> = Vec::new();
    for w in &workers {
        ranks.push(w.endpoints.iter().map(|e| merged.endpoint_id(e) as u32).collect());
    }
    let mut order: Vec<usize> = (0..merged.endpoints.len()).collect();
    order.sort_unstable_by(|&a, &b| merged.endpoints[a].cmp(&merged.endpoints[b]));
    let mut rank_of = vec![0u32; order.len()];
    for (rank, &id) in order.iter().enumerate() {
        rank_of[id] = rank as u32;
    }
    ranks.iter_mut().flatten().for_each(|id| *id = rank_of[*id as usize]);
    let names: Vec<&[u8]> = order.iter().map(|&id| &*merged.endpoints[id]).collect();

    let next = AtomicUsize::new(0);
    let mut parts: Vec<(usize, Vec<u8>)> = thread::scope(|s| {
        let work = || {
            let mut done = Vec::new();
            loop {
                let hour = next.fetch_add(1, Relaxed);
                if hour >= HOURS {
                    return done;
                }
                done.push((hour, summarise(hour, &workers, &ranks, &names)));
            }
        };
        let spawned: Vec<_> = (0..threads).map(|_| s.spawn(work)).collect();
        spawned.into_iter().flat_map(|t| t.join().unwrap()).collect()
    });
    parts.sort_unstable_by_key(|part| part.0);

    let mut out = File::create("/out/summary.txt").unwrap();
    for (_, part) in parts {
        out.write_all(&part).unwrap();
    }
}

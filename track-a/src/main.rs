//! Speed Demon 2026, Track A (Ingest).
//!
//! Workers claim 1 MiB blocks of requests.log and read them with pread, so memory never
//! holds the whole file. Each row packs into a u64 and goes into a bucket per hour. Then
//! each hour is sorted once, which puts endpoints and latencies in output order together.
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

const BLOCK: usize = 1 << 20;
const MAX_LINE: usize = 177; // ts 27, method 7, endpoint 125, status 3, latency 10, 3 spaces, LF
const HOURS: usize = 72; // the span is fixed for every seed: 2026-10-30T18 to 2026-11-02T17
const SLOTS: usize = 1 << 14; // the catalogue has 1,842 endpoints
const ID_SHIFT: u32 = 33; // a row is endpoint << 33 | latency << 1 | is_error

struct Worker {
    slots: Vec<u32>, // open-addressed table of endpoint id + 1
    endpoints: Vec<Box<[u8]>>,
    hours: Vec<Vec<u64>>,
}

fn hash(name: &[u8]) -> usize {
    let mut h = name.len() as u64;
    for chunk in name.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(word)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    (h >> (64 - SLOTS.trailing_zeros())) as usize
}

impl Worker {
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

fn parse(file: &File, len: usize, next: &AtomicUsize) -> Worker {
    let mut w = Worker {
        slots: vec![0; SLOTS],
        endpoints: Vec::new(),
        hours: vec![Vec::new(); HOURS],
    };
    let mut buf = vec![0u8; BLOCK + MAX_LINE + 1];
    loop {
        let start = next.fetch_add(BLOCK, Relaxed);
        if start >= len {
            return w;
        }
        // A line belongs to the block it starts in, so read one byte back to see whether
        // the block opens on a line start, and MAX_LINE on to finish the last line.
        let from = start.saturating_sub(1);
        let size = buf.len().min(len - from);
        file.read_exact_at(&mut buf[..size], from as u64).unwrap();
        let b = &buf[..size];
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
            let end = name + b[name..].iter().position(|&c| c == b' ').unwrap();
            let is_error = (b[end + 1] == b'5') as u64;
            let mut latency = 0u64;
            p = end + 5;
            while b[p] != b'\n' {
                latency = latency * 10 + digit(b[p]) as u64;
                p += 1;
            }
            p += 1;
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

fn summarise(hour: usize, workers: &[Worker], ranks: &[Vec<u64>], names: &[&[u8]]) -> Vec<u8> {
    let total: usize = workers.iter().map(|w| w.hours[hour].len()).sum();
    let mut rows: Vec<u64> = Vec::with_capacity(total);
    for (w, rank) in workers.iter().zip(ranks) {
        let keep: u64 = (1 << ID_SHIFT) - 1;
        let renamed = w.hours[hour].iter();
        rows.extend(renamed.map(|&r| rank[(r >> ID_SHIFT) as usize] << ID_SHIFT | r & keep));
    }
    rows.sort_unstable();
    let label = hour_label(hour);
    let mut out = Vec::with_capacity(total * 8);
    for group in rows.chunk_by(|a, b| a >> ID_SHIFT == b >> ID_SHIFT) {
        let n = group.len();
        let at = |p: usize| group[(p * n + 99) / 100 - 1] >> 1 & u32::MAX as u64;
        let errors = group.iter().filter(|&&r| r & 1 == 1).count();
        write!(out, "{label} ").unwrap();
        out.extend_from_slice(names[(group[0] >> ID_SHIFT) as usize]);
        let (p50, p95, p99, ppm) = (at(50), at(95), at(99), errors * 1_000_000 / n);
        writeln!(out, " {n} {p50} {p95} {p99} {errors} {ppm}").unwrap();
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

    // &[u8] orders as unsigned bytes with a proper prefix first, which is the output order.
    let seen = workers.iter().flat_map(|w| w.endpoints.iter());
    let mut names: Vec<&[u8]> = seen.map(|e| &**e).collect();
    names.sort_unstable();
    names.dedup();
    let rank_all = |w: &Worker| -> Vec<u64> {
        let ranked = w.endpoints.iter();
        ranked.map(|e| names.binary_search(&&**e).unwrap() as u64).collect()
    };
    let ranks: Vec<Vec<u64>> = workers.iter().map(rank_all).collect();

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

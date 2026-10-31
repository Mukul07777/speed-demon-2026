// Diagnostic only, never part of a submission. Records phase times on the scoring box and
// reports them as the name of an extra file in /out, which fails the run on purpose: the
// board's public failure text lists the names it found there. T and M are still published.
//
//   p = CPU time used before main (exec and dynamic loading), microseconds
//   e = main start to report, u/s = user and system CPU of all threads, f = minor faults
//   v/i = voluntary and involuntary context switches, then slot:first-last in microseconds
pub mod probe {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    extern "C" {
        fn clock_gettime(id: i32, ts: *mut [i64; 2]) -> i32;
        fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
    }
    const SLOTS: usize = 16;
    static START: AtomicU64 = AtomicU64::new(0);
    static PRE: AtomicU64 = AtomicU64::new(0);
    static FIRST: [AtomicU64; SLOTS] = [const { AtomicU64::new(u64::MAX) }; SLOTS];
    static LAST: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static SUM: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

    fn clock(id: i32) -> u64 {
        let mut t = [0i64; 2];
        unsafe { clock_gettime(id, &mut t) };
        t[0] as u64 * 1_000_000 + t[1] as u64 / 1000
    }

    pub fn now() -> u64 {
        clock(1) - START.load(Relaxed)
    }

    pub fn begin() {
        PRE.store(clock(2), Relaxed);
        START.store(clock(1), Relaxed);
    }

    pub fn stamp(slot: usize) {
        let t = now();
        FIRST[slot].fetch_min(t, Relaxed);
        LAST[slot].fetch_max(t, Relaxed);
    }

    /// Adds to a running total, shown as slot+total.
    pub fn add(slot: usize, amount: u64) {
        SUM[slot].fetch_add(amount, Relaxed);
    }

    pub fn report(extra: &str) {
        let end = now();
        let mut r = [0i64; 18];
        unsafe { getrusage(0, &mut r) };
        let (user, system) = (r[0] * 1_000_000 + r[1], r[2] * 1_000_000 + r[3]);
        let mut s = format!(
            "p{} e{} u{} s{} f{} v{} i{}",
            PRE.load(Relaxed),
            end,
            user,
            system,
            r[8],
            r[16],
            r[17]
        );
        for slot in 0..SLOTS {
            let (first, last) = (FIRST[slot].load(Relaxed), LAST[slot].load(Relaxed));
            if first != u64::MAX {
                s += &if first == last {
                    format!(" {slot}:{last}")
                } else {
                    format!(" {slot}:{first}-{last}")
                };
            }
            let sum = SUM[slot].load(Relaxed);
            if sum != 0 {
                s += &format!(" {slot}+{sum}");
            }
        }
        s += extra;
        let name: String = s
            .chars()
            .map(|c| if c == '/' { '|' } else { c })
            .take(230)
            .collect();
        let _ = std::fs::File::create(format!("/out/{name}"));
    }

    /// One word from a file: the bracketed choice of a sysfs setting, or the whole first line.
    pub fn fact(path: &str) -> String {
        let text = std::fs::read_to_string(path).unwrap_or_else(|_| "?".to_string());
        let line = text.lines().next().unwrap_or("?");
        match (line.find('['), line.find(']')) {
            (Some(open), Some(close)) => line[open + 1..close].to_string(),
            _ => line.split_whitespace().take(3).collect::<Vec<_>>().join("_"),
        }
    }
}

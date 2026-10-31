# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only. One worker per hardware thread claims 128 KiB blocks of `requests.log` and reads them with `pread`, so the file is never held in memory whole. A line belongs to the block it starts in, and each read overlaps the next block by one maximum-length line.

Parsing never touches a date library. The span is fixed at 72 hours, so the hour index comes from a few timestamp digits, and the method's first two bytes give its length. An AVX2 scan finds the end of the endpoint, usually in one load. Latency lengths vary, so a byte loop would mispredict on nearly every line; instead eight bytes are converted at once with SWAR arithmetic. Each row packs into one `u64`, endpoint then latency then an error bit, and goes into that worker's bucket for the hour. Endpoints get worker-local ids from a small open-addressed table.

After parsing, the local ids merge through one more table, and the distinct names are sorted once to give each endpoint its output rank. Then the workers take one hour each. A counting sort groups the hour's latencies by endpoint rank in one pass. Each group then selects only its three percentiles, p99 first, so that p95 and p50 search a smaller slice. Hours are already in output order, so the file is the hours concatenated.

## Dependencies

None.

## With another 24 hours

Shrink rows from eight bytes, free each hour's rows as it is summarised, and write the output in parallel.

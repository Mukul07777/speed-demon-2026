# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

Rust, standard library only. One worker per hardware thread claims 1 MiB blocks of `requests.log` and reads them with `pread`, so the file is never held in memory whole. A line belongs to the block it starts in, and each read overlaps the next block by one maximum-length line.

Parsing never touches a date library. The span is fixed at 72 hours, so the hour index comes from a few timestamp digits, and the method's first two bytes give its length. Each row packs into one `u64`, endpoint then latency then an error bit, and goes into that worker's bucket for the hour. Endpoints get worker-local ids from a small open-addressed table.

Then the workers take one hour each: map local ids to the endpoint's sorted rank, sort the hour's rows once, and read every group's count, errors and nearest-rank percentiles straight from the sorted run. Sorting the whole key means the same sort gives both the output order and the percentiles. Hours are already in output order, so the file is the hours concatenated.

## Dependencies

None.

## With another 24 hours

Replace the byte-at-a-time newline and space scans with SIMD, swap `sort_unstable` for a radix sort on the packed rows, and use selection instead of a full sort for large groups.

# V4.1 Engram residency: the page cache does not keep the hot set; a small row cache would

M5 Ultra (`metagross`), 2026-11-13. Branch `research/v41-engram-residency`,
based on main `56a1e8fe5`. Measurement only: no product change. The only
source change is an off-by-default `QWEN_V41_ENGRAM_TRACE` row tap plus
offline analysis and residency scripts.

## Decision

**Do not pre-warm or add a page-cache keep-hot hint. Price a bounded
in-process Engram row cache (option b) next.**

- The 10-minute greedy window touches **1.58-1.60M distinct rows** and
  **1.40-1.41M distinct 16 KiB pages (23.0-23.1 GB)**. The 90% hot set is
  **0.89-0.91M rows and 0.82-0.84M pages (13.4-13.7 GB)**.
- The same rows cost only **0.23 GB packed (144 B/row)** or **1.6 GB as F32
  (1 KiB/row)**. The page footprint is 100x the packed footprint because a
  144-byte row faults a 16 KiB page.
- The OS retained only **4.7-7.7 GB** of Engram pages across the two windows
  (window min 1.8-2.6 GB, max 5.7-10.5 GB), despite ~55-59 GB of
  free+inactive+speculative memory. It re-read the same distinct pages about
  **3.4x** (cold) and **2.7x** (warm).
- A true purged cold window read **77.45 GB**; the back-to-back warm window
  read **62.57 GB (-19.2%)** but ran only **+1.45%** faster (79.76 -> 80.92
  tok/s). Both were thermal max 1. The measured anchor is therefore ~1.5%
  for a 19% read cut; the QoS disk-rate slope bounds a near-complete cut at
  ~7%. Expected row-cache gain: **~4-7%**, not the full 10%.
- Pre-warm only repopulates pages the OS then evicts, and adds 100-170 s of
  startup. `F_RDADVISE` is already used; macOS has no supported keep-hot
  advice, and wiring 13-23 GB would spend the headroom.

## Access pattern

`QWEN_SERVE_V41_PREFILL_TIMING=1` gives prefill interval counters but no
row identity. The added `QWEN_V41_ENGRAM_TRACE=<prefix>` writes two fixed
files from `DiskEngram::read_rows_with`: `<prefix>.rows` is 9-byte
`(layer u8, row u64)` records, and `<prefix>.times` is 16-byte
`(epoch_ns, rows_written)` call boundaries. The write sits before the timed
read loop, so `engram_read_wall_ms` is unchanged. The tap is compiled in but
inert unless the variable is set.

Three-agent 48K plan, seed `20261109`. Step 1 is the standard 180 s plan with
agent-default sampling; steps 3 are 600 s greedy windows after a targeted
`msync(MS_INVALIDATE)` of the Engram file (`step3-cold`), followed
immediately by the same plan without the purge (`step3-warm`). Measurement
window only; warm-up reads are excluded.

| arm | tokens | accesses | distinct rows | distinct pages | page bytes | mean reuse | singleton rows | top 1% access share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step1 agent 180 s | 14,542 | 1,547,520 | 600,157 | 573,552 | 8.75 GiB | 2.58 | 76.7% | 15.8% |
| step3-cold 600 s | 47,858 | 6,888,000 | 1,580,995 | 1,395,318 | 21.29 GiB | 4.36 | 56.8% | 23.5% |
| step3-warm 600 s | 48,550 | 6,938,544 | 1,600,221 | 1,411,432 | 21.54 GiB | 4.34 | 55.6% | 23.7% |

Hot-set coverage (rows, packed bytes, 16 KiB pages, resident page bytes):

| arm | coverage | distinct rows | packed bytes | pages | page bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 90% | 892,195 | 0.128 GB | 820,911 | 12.53 GiB |
| cold | 95% | 1,236,595 | 0.178 GB | 1,111,765 | 16.96 GiB |
| cold | 99% | 1,512,115 | 0.218 GB | 1,339,738 | 20.44 GiB |
| warm | 90% | 906,367 | 0.131 GB | 836,230 | 12.76 GiB |
| warm | 95% | 1,253,294 | 0.180 GB | 1,123,186 | 17.14 GiB |
| warm | 99% | 1,530,836 | 0.220 GB | 1,349,801 | 20.60 GiB |

The full 10-minute working set is **1.6 GB as F32**, or **0.23 GB packed**.
A 2-4 GB cache with hash/LRU overhead covers everything the window touched;
a 1 GB F32 cache covers the 90% hot set.

Growth is sublinear and event-shaped, because large agent appends arrive
between decode stretches. Distinct rows per 1k delivered tokens in the cold
window: 51.2 at 60 s, 43.2 at 120 s, 38.5 at 180 s, 35.6 at 240 s, 44.4 at
300 s (a large append), 34.8 at 540 s, **33.0 at 600 s**. Fitted slope
**32.4 rows/token** (cold) and **33.2** (warm). In page-cache terms this is
**~0.47 MB of new resident pages per served token**, or ~29 pages/token; the
packed row growth is only ~4.8 KB/token.

## Page cache and disk

Disk is the sum of `process_token_io` + `process_prefill_io`
`process_disk_read_bytes` for calls that first-token inside the window.
Prefill rows and read latency come from the per-call prefill timing records;
"cached share" uses 0.5 us (hit) and 65 us (miss) per row. Residency is
`mincore(2)` on the Engram file, sampled every 10 s.

| arm | window disk | rate | prefill disk | decode disk | prefill rows | prefill us/row | cached share | decode rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step3-cold | 77.45 GB | 129.1 MB/s | 15.72 GB | 61.73 GB | 2,628,720 | 3.37 | 95.6% | 4,259,280 |
| step3-warm | 62.57 GB | 104.3 MB/s | 6.22 GB | 56.35 GB | 2,672,592 | 2.22 | 97.3% | 4,265,952 |

| arm | Engram resident at window start | at window end | window min | window max | disk bytes / access |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 4.69 GB (286,147 pages) | 5.06 GB (308,592) | 1.76 GB (107,201) | 5.73 GB (349,649) | 11,245 |
| warm | 7.68 GB (468,885 pages) | 7.47 GB (456,188) | 2.62 GB (159,566) | 10.48 GB (639,404) | 9,018 |

With three 48K sessions, `vm_stat` at window start showed **164.8 GB wired**,
**6.7-6.9 GB free**, **46.4-49.0 GB inactive**, **2.9-5.6 GB speculative**,
and **28-31 GB file-backed**. The free+inactive+speculative pool is
**~59 GB**, consistent with the ~57 GB fit headroom, but the Engram file only
held 4.7-7.7 GB. The purge proves the cold start: residency was **0 pages**
after `msync`, and the first warm-up TTFT rose to 26-36 s (from ~1 s warm),
although total warm-up stayed ~96 s because the 48K chunks overlap.

The warm arm cut prefill reads by 60% but decode reads by only 9%. Decode is
the residency problem: 4.26M decode rows per 600 s, 56-62 GB of reads, and
only ~7 GB of retained pages. The page cache is not preferentially keeping
clean Engram pages even when ~59 GB is reclaimable.

## Pricing

QoS greedy arms from `2026-11-12-serve-qos-nice0.md` order by disk reads:
1.72 GB / 100.644 tok/s, 7.41 / 100.394, 18.56 / 96.411, 23.83 / 90.167 over
180 s. A linear fit gives **-0.448 tok/s per GB** (r2 0.858), or
**-0.081 tok/s per MB/s**. The fit is flat below ~40 MB/s and steepens above
~100 MB/s; it is an upper bound when extrapolated to a 600 s window.

| option | cost | expected gain | read |
| --- | --- | ---: | --- |
| (a) pre-warm hot set at start | read 13.4 GB (90%) / 22.0 GB (99%) at 129 MB/s = 104-170 s startup; no pinning | ~1% or less | Pages are evicted by the same window; the measured warm-start gain is only 1.45% for 19% fewer reads |
| (b) in-process row cache, 2-4 GB F32 (or 0.5-1 GB packed) | 1.6 GB F32 for the full 10-min working set; 2-4 GB with indexing; LRU/sharded; deterministic rows | **4-7%** | Removes ~50 GB of the 56-62 GB decode re-reads; measured 1.45% per 19% cut, QoS slope ~7% near-complete |
| (c) `madvise` / `F_RDADVISE` keep-hot | `F_RDADVISE` already does read-ahead; macOS has no keep-hot advice; `mlock` would wire 13-23 GB | low / negative | Wiring spends the 57 GB headroom and risks pressure; `msync` shows clean pages are reclaimable |
| (d) nothing | 0 | 0 | Leaves 56-77 GB per 10 min and ~1.5-7% on the table |

**Recommendation: prototype (b) with a 2-4 GB F32 cache (or 1 GB packed),
keyed by `(layer,row)`, before any page-cache hint work.** Run a 48K
three-agent A/B; the acceptance signal is a decode disk-read cut of at least
50% and a paired tok/s gain outside host noise. The cache is safe because
Engram rows are immutable, and 2-4 GB is noise against 59 GB of reclaimable
memory. First-touch reads remain: a cold session still pays one page read per
distinct row (~22-23 GB at 16 KiB), so do not price the first session at the
full 7%.

## Limits

- The trace writes 9 bytes per row and flushes per read call. It sits outside
  `engram_read_wall_ms`, but it is not a zero-cost arm; absolute tok/s is a
  guide, not a clean product quote.
- The greedy pair is not bit-identical: 75 of 90 common completed calls have
  the same response digest, first divergence `a1-t2`; window tokens differ by
  1.4%. The disk difference is the controlled variable.
- Hot-set membership is session-specific. These numbers show within-session
  reuse; they do not prove cross-session overlap. A cache that learns online
  must be priced on a second-session hit rate.
- Page-cache residency is `mincore` on the file, not per-page access time. The
  16 KiB page cost is the allocation granularity; physical read bytes are
  measured separately and can be smaller for contiguous extents.

## Records

- Branch `research/v41-engram-residency` at `2953dc7c` from main
  `56a1e8fe5`; not merged. Build `qwen` `851b840e73f4`, `qwen-tok`
  `1e45037f288d` (sha256 prefixes).
- Every build and arm ran under `QWEN_METAL_LEASE_WAIT=1
  scripts/bench/with-metal-lease --task v41-engram-res --for 15m`; build at
  nice 10, every server and harness at nice 0 (`ps -o nice=` sampled in
  `serve/*.server-nice.txt`). The purged cold arm held 14m48s and the warm arm 14m09s (target 15m).
- Thermal: settle to <=1 for 60 s, allow 2, abort at 3. Step1 max 1;
  step3-cold and step3-warm max 1 throughout their windows; no arm aborted.
- Raw: `metagross:~/qwen-runs/2026-11-13-v41-engram-residency/`, including
  `plan-180.json`, `plan-600.json`, `serve/{step1-agent,step3-cold,step3-warm}/`
  (`trace.rows`, `trace.times`, `trace-window.json`, `calls.jsonl`,
  `native-lines.json`, `summary.json`), `*.residency.jsonl`, and
  `logs/step3-cold.invalidate.jsonl` (135,209 -> 0 resident pages).
- Preliminary unpurged pair (step2) is in the same directory: cold 68.54 GB /
  80.43 tok/s (thermal max 2) and warm 55.21 GB / 85.87 tok/s (thermal max 1).
  It is not the primary comparison because the cold arm inherited Step 1's
  page cache and the thermal states differ.

# V4.1 Engram row cache: exact, default-on, native disk cut; serve gain inside host noise

M5 Ultra (`metagross`), 2026-11-13. Branch `perf/v41-engram-row-cache`
from main `56a1e8fe5`. Product change: a bounded in-process cache of packed
Engram rows. Native decode is bitwise and 3.8-4.9% faster with 99.4% fewer
physical decode reads. Three interleaved three-agent 48K serve pairs point
+4.6% but the paired 95% CI is wide and every window overlapped unrelated
compiler load.

## Implementation

`DiskEngram::read_rows_with` now checks a process-wide (per resident model)
`(layer, row)` cache before `pread`. A hit copies the cached packed bytes into
the existing staging buffer and decodes through the same
`dequant_to_f32_in_place` call, so values are identical by construction.
`advise_rows` skips rows already resident.

- `crates/qwen-llm/src/deepseek_v41/gguf/engram_cache.rs`: CLOCK
  (second-chance) eviction, capacity counted in packed payload bytes, one
  `Box<[u8]>` per row.
- Default `QWEN_V41_ENGRAM_ROW_CACHE_MB=1024`; `0` disables it. One instance
  lives in `ResidentWeights`, so all sessions in one `qwen serve` process
  share it.
- `QWEN_V41_ENGRAM_ROW_CACHE_STATS=1` prints capacity at construction and
  packed bytes, rows, hits and misses at teardown. Default off.
- The server's single resident `DiskEngram` is the sharing point; no global
  mutable cache or cross-model key.

Tests: four CLOCK unit tests (hit bytes, capacity bound, second chance,
oversized row) and one CPU integration test comparing uncached, cached and
batched reads and asserting the two-row capacity bound. No ignored tests.

## Native decode probe

Same `iter` binary, 3-wide independent decode, 512 rounds, 4096-row suffix
after a CED root. The Engram companion is purged with
`msync(MS_INVALIDATE)` after warm-up, immediately before the measured loop,
so both arms start with zero resident Engram pages. All full-vocabulary
hashes are identical between arms at both depths.

| depth | cache | ms/round | ms/token | aggregate tok/s | decode disk | bytes/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | on | **28.242** | **9.414** | **106.225** | **4,734,976** | **3,083** |
| 32K | `=0` | 29.620 | 9.873 | 101.282 | 773,701,632 | 503,712 |
| 48K | on | **27.966** | **9.322** | **107.275** | **4,734,976** | **3,083** |
| 48K | `=0` | 29.058 | 9.686 | 103.241 | 773,701,632 | 503,712 |

Native wall gain **+4.88%** at 32K and **+3.76%** at 48K; physical decode
reads **-99.4%** in both. The on arm reads only the distinct misses; the off
arm pays a 16 KiB page for repeated rows.

### Landing witness: three interleaved 48K pairs

Landed tree `77224e49`, `iter` profile, one binary, 48K 3-wide 512-round.
Order is ABBAAB (A off, B default); each arm loaded the model, ran
`invalidate_file_cache.py` on the Engram companion, then let the probe purge
again immediately before the measured loop (resident pages after: 0). Every
arm ran at niceness 0 (`ps -o nice=` recorded 0) under the metal lease.

| arm | cache | ms/round | decode disk bytes |
| --- | --- | ---: | ---: |
| A1 | `=0` | 35.213 | 773,701,632 |
| B1 | default | 27.121 | 4,734,976 |
| B2 | default | 28.264 | 4,734,976 |
| A2 | `=0` | 30.231 | 773,701,632 |
| A3 | `=0` | 29.396 | 773,701,632 |
| B3 | default | 28.081 | 4,734,976 |

Off/on ms-per-round ratios: pair 1 **1.298**, pair 2 **1.070**, pair 3
**1.047**; mean **1.136** (+12.0% on arm). All endpoint hashes are identical
across all six arms. Raw `metagross:~/qwen-runs/2026-11-13-engram-cache-landing/`;
probe sha256 `cd33e366a2c25728e567e6ee0c48be447ee09762070666beae95e33fb5242ee7`.

## Serve A/B

`scripts/bench/serve_realistic.py`, three warm agents, 49,152-token roots,
196,608 context, agent-default sampling, plan seed `20261109`, 15 s ramp,
180 s windows, `max_active=4`, three sampler workers, fair prefill. Every arm
was purged with `msync(MS_INVALIDATE)` (resident pages after: 0), ran at
niceness 0 (harness and server), and settled at pressure <=1 for 60 s before
load; pressure 2 was allowed during warm-up/window and 3 aborts. Order is
ABBAAB: A1 off, B1 on, B2 on, A2 off, A3 off, B3 on.

| arm | cache | tok/s | ITL p99 ms | append TTFT p50 / p95 ms | decode disk GB | thermal max |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| A1 | `=0` | 79.500 | 177.6 | 948 / 2509 | 7.383 | 1 |
| B1 | default | 90.133 | 55.8 | 911 / 1730 | 7.048 | 1 |
| B2 | default | 80.239 | 152.8 | 964 / 2947 | 7.412 | 1 |
| A2 | `=0` | 81.411 | 150.8 | 939 / 1918 | 12.712 | 0 |
| A3 | `=0` | 82.344 | 156.3 | 960 / 1896 | 8.223 | 2 |
| B3 | default | 83.783 | 120.9 | 989 / 1810 | 9.662 | 1 |

Paired relative tok/s: **+13.38%, -1.44%, +1.75%**; mean **+4.56%**, paired
Student-t 95% CI **[-14.81%, +23.93%]** (df 2). Absolute mean **+3.63 tok/s**,
CI **[-11.77, +19.04]**. ITL p99 is lower in 2 of 3 pairs (mean 109.8 vs
161.6 ms) and append TTFT is flat. Window disk is the sum of
`process_token_io` + `process_prefill_io` `process_disk_read_bytes` for calls
that first-token inside the window.

A diagnostic default-on arm with stats recorded **6,326,318 hits and
2,758,594 misses (69.6%)**, 2,758,594 cached rows, 397,237,536 packed bytes of
the 1,073,741,824-byte capacity. The cache is active and covers the window.

## Interpretation

The cache removes about 70% of logical decode row reads. In this 180 s serve
window that cut physical disk bytes only in the A2/B2 pair: most of the
remaining 2.76M misses are compulsory pages that the off arm also had to read
once, and the OS page cache was already absorbing many repeats. A longer
window or a smaller OS cache would price the eviction saving more cleanly.
The native probe, where the 16 KiB page cost dominates 73K accesses, shows the
mechanism at its ceiling.

Every serve window overlapped compiler processes from other lanes and host
load was 12-37, so the tok/s interval, not the point estimate, is the honest
serve result. Pressure 2 appears in A3 (50 s of its window) and in B2 before
its window; the three pairs are not thermally equal. No arm aborted.

## Records

- Branch `perf/v41-engram-row-cache`, base `56a1e8fe5`; not merged.
- Serve arms used release `qwen` sha256 `1ab6cc8aad6dcb7142741e2d45df21a572ca99c4492ef7df96de541871139068`,
  `qwen-tok` `f9b435e4c0a8ad1e1540a898c312d26a9a2da88b83a52c38e8e0316f5dc0d026`.
  The diagnostic arm used `bc5cd0a8f15b86dfc1b16b98325f0a02ad950b136be1cd923099a470dde8b772`,
  which differs only by the inert stats counters. Native probe `2225f64a7b17605fbdac9cf8016bdb5eedddada15ddb8e22b7206ed0b009e7fa`.
- Every build and arm ran under `QWEN_METAL_LEASE_WAIT=1
  scripts/bench/with-metal-lease --task v41-engram-cache --for 15m`;
  measurements at nice 0. No other model process was resident at load.
- Raw: `metagross:~/qwen-runs/2026-11-13-v41-engram-cache/`, including
  `plan-180.json`, `build.sha256`, `native/`, `serve/{a1-off,b1-on,b2-on,a2-off,a3-off,b3-on,diag-on}/`,
  `logs/*.invalidate.jsonl`, `*.server-nice.txt`, and `analyze.py`.

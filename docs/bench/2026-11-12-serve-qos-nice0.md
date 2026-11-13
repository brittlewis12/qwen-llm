# Serve QoS at normal priority: no throughput gain survives; append TTFT stays lower

**Question.** The 2026-11-12 QoS coverage gate measured **+10.72%** tok/s for
V4.1 three-agent serve, but its arm script launched both
`serve_realistic.py` and `qwen serve` under `nice -n 10`. Under the M5's
background load a deprioritized server gains more from a QoS raise than a
nice-0 server would. Does the landed change (`3d831eacd`, on main since
`1edb7e8d8`) still help when the server runs at normal CPU priority, as
production does?

**Answer: no reliable throughput gain.** On current main **`5ee5c8c4f`**,
three interleaved agent-sampling pairs give **+6.44% tok/s**, 95% CI
**[-12.14%, +25.02%]**, df2. The interval includes zero, and the arm changes
track host load and process disk reads, not the QoS flag. A deterministic
greedy same-token pair is **-4.21%** (default on slower), with the on arm
reading **18.56 GB** from disk versus **1.72 GB** for the rollback. Across the
four low-load greedy arms, throughput orders almost exactly by disk reads
(1.7 GB -> 100.6 tok/s, 7.4 -> 100.4, 18.6 -> 96.4, 23.8 -> 90.2). The one
signal that survives at nice 0 is **append TTFT p95**: -19.79%, 95% CI
[-31.67%, -7.92%], lower in every agent-sampling pair. The original +10.72%
should be read as a nice-10 result, not a production-priority throughput
claim.

## Protocol and build

One release build of `qwen` and `qwen-tok` from `5ee5c8c4f` in a fresh M5
worktree. Binary sha256 prefixes: `qwen` **bf76b2d6a085**, `qwen-tok`
**6865eadb1582**. The build ran under the canonical Metal lease with
`nice -n 10`; no serve or harness process was niced.

`scripts/bench/serve_realistic.py run`, three agents, the original gate's
plan (`plan.json`, seed **20261109**, family `v41`), 49,152 warm tokens per
client, context 196,608, agent-default sampling, 15 s ramp, 180 s measurement
window, `QWEN_SERVE_V41_MAX_ACTIVE=4`, `QWEN_SERVE_V41_SAMPLE_WORKERS=3`,
`QWEN_SERVE_V41_PREFILL_FAIR=1`. Each arm settled to thermal pressure <= 1 for
60 s inside its own lease before load and after warm-up; pressure 2 was allowed
during the window and 3 would abort. No arm aborted. Every arm checked that no
other model process was resident before loading.

Arms: **A = default QoS (on)**, **B = `QWEN_GPU_THREAD_QOS=0`**. Order
**A1 B1 B2 A2 A3 B3**, pairs `(A1,B1)`, `(A2,B2)`, `(A3,B3)`.

## Niceness control

The first launch ran the gate script as a zsh background job. zsh's `BGNICE`
made that script nice 5, and the first A1 server inherited nice 5. That arm
and the aborted B1 host snapshot were discarded under
`serve-invalid-nice5/`; the scripts now refuse to run at any niceness other
than 0, and the gate was relaunched in the foreground. Every reported arm
records `harness_nice=0` and `server_nice=0` in `serve/<arm>.server-nice.txt`
via `ps -o nice=`; all ten arms record server nice 0 (one or more samples each).

## V4.1 agent-sampling gate

| arm | QoS | tok/s | ITL p99 (s) | append TTFT p95 (s) | thermal max | window tokens | disk read (GB) | response calls / digest | load before -> during mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A1 | on | 79.511 | 0.0567 | 2.224 | 2 | 14,312 | 10.78 | 37 / `862655782f3429cf` | 17.2 -> 31.3 / 46.0 |
| B1 | off | 80.767 | 0.0349 | 2.615 | 1 | 14,538 | 9.30 | 32 / `fcf9ae830f8f9ceb` | 45.7 -> 18.5 / 21.3 |
| B2 | off | 80.067 | 0.0358 | 2.825 | 1 | 14,412 | 7.98 | 33 / `722f9f5ff3f0829e` | 18.3 -> 56.4 / 77.8 |
| A2 | on | 86.150 | 0.0331 | 2.132 | 2 | 15,507 | 7.84 | 25 / `af8558a4ca165cbe` | 26.1 -> 23.0 / 30.8 |
| A3 | on | 95.700 | 0.0318 | 1.853 | 0 | 17,226 | 13.02 | 27 / `04979378a7bd4a2f` | 28.8 -> 4.4 / 5.3 |
| B3 | off | 84.489 | 0.0532 | 2.314 | 0 | 15,208 | 23.93 | 41 / `23213a8c2bee4809` | 4.6 -> 4.0 / 4.4 |

Load is the 1-minute load average before the arm and the mean/max of 10 s
samples inside it. Disk read is the sum of per-call
`process_token_io` + `process_prefill_io` `process_disk_read_bytes` totals.
Response digest is `sha256` over per-call completed-response text; full
per-call hashes are in `response-hashes.json`.

| pair | temporal order | tok/s change (on vs off) | ITL p99 ratio (on/off) | worst-gap change | TTFT p95 change |
| --- | --- | ---: | ---: | ---: | ---: |
| B1/A1 | A then B | -1.55% | 1.624 | +0.465 s | -14.96% |
| B2/A2 | B then A | +7.60% | 0.925 | +0.556 s | -24.52% |
| B3/A3 | A then B | +13.27% | 0.598 | -0.336 s | -19.89% |

Paired Student-t over three arm pairs (df 2): tok/s relative mean **+6.44%**,
95% CI **[-12.14%, +25.02%]**; absolute mean **+5.35 tok/s**,
**[-10.22, +20.91]**. ITL p99 ratio mean **1.049** (CI [-0.254, 2.351]);
worst-gap mean **+0.228 s** (CI [-0.991, +1.448]); append TTFT p95 relative
mean **-19.79%** (CI **[-31.67%, -7.92%]**).

**Reading.** The throughput interval includes zero and its three pair changes
span 15 percentage points. The arms that ran at load 4-5 were 14-20% faster
than the arms that ran at load 23-56, and B3 read 23.9 GB while A3 read
13.0 GB. The mean is therefore not a clean QoS price. The TTFT p95 direction
is different: all three pairs improve with QoS on, and the interval excludes
zero.

## Greedy same-token control

Because the agent result is near 5%, two greedy pairs were added, in opposite
orders, with the same plan and settings (`--sampling greedy`; temperature 0,
top-p 1). G1/G2 is on-then-off; G3/G4 is off-then-on.

| arm | QoS | tok/s | ITL p99 (s) | append TTFT p95 (s) | thermal max | window tokens | disk read (GB) | response calls / digest | load before -> during mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| G1 | on | 90.167 | 0.0360 | 3.720 | 0 | 16,230 | 23.83 | 31 / `f334d62042c47053` | 4.0 -> 4.2 / 5.0 |
| G2 | off | 100.394 | 0.0301 | 1.776 | 0 | 18,071 | 7.41 | 24 / `9e3b0fbc2b8bff40` | 4.5 -> 4.0 / 5.0 |
| G3 | off | 100.644 | 0.0301 | 1.777 | 0 | 18,116 | 1.72 | 24 / `18bdcf24841cd4cd` | 3.9 -> 4.4 / 5.0 |
| G4 | on | 96.411 | 0.0282 | 2.694 | 0 | 17,354 | 18.56 | 23 / `fc33cfeb56c67203` | 4.6 -> 4.2 / 5.2 |

| pair | temporal order | tok/s change (on vs off) | ITL p99 ratio (on/off) | identical common completed calls |
| --- | --- | ---: | ---: | --- |
| G2/G1 | on then off | -10.19% | 1.196 | 20 / 21 |
| G3/G4 | off then on | -4.21% | 0.938 | **23 / 23** |

The G3/G4 pair is the deterministic same-token control: every completed call
common to both arms has the same token hash, and the on arm is 4.21% slower.
It is not an I/O-controlled pair: G4 read 18.56 GB while G3 read 1.72 GB. In
both greedy pairs the on arm was the cold one. The four low-load greedy arms
order by disk reads rather than by the QoS flag: 1.72 GB / 100.644 tok/s,
7.41 / 100.394, 18.56 / 96.411, 23.83 / 90.167. A two-pair greedy mean is
-7.20% with a df1 interval [-45.20%, +30.80%], which is too wide to price
anything. The G1/G2 pair has one greedy decode divergence at `a1-t3`; the
G3/G4 pair does not.

## Conclusion

At normal CPU priority the V4.1 three-agent throughput gain does not reproduce
in a way that can be separated from host load and page-cache state. The
agent-sampling mean is +6.44% with an interval that includes zero, and the
same-token greedy pair is -4.21% with the on arm reading ten times more disk.
The original +10.72% remains valid for the gate that ran the whole server under
`nice -n 10`, but it is not evidence of a production-priority throughput win.
The append TTFT p95 improvement is the only nice-0 signal that is consistent
across pairs (-19.79%, CI [-31.67%, -7.92%]); a throughput claim would need an
isolated or page-cache-controlled host.

## Records

- Raw packet: `metagross:~/qwen-runs/2026-11-12-qos-nice0/`, including
  `serve/{A1,B1,B2,A2,A3,B3,G1,G2,G3,G4}/summary.json`, `calls.jsonl`,
  retained event files, per-arm `host-before.txt`, `host-after.txt`,
  `*.server-nice.txt`, `*.load-during.txt`, `response-hashes.json`,
  `analysis-main.json`, `analysis-greedy.json`, and the gate logs.
- Discarded niceness-5 packet: `serve-invalid-nice5/` in the same run
  directory.
- Branch `bench/qos-nice0` from main `5ee5c8c4f`; not merged. The QoS change
  `3d831eacd` is unchanged.
- Every build and arm ran under
  `QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease`; one arm per hold,
  each under 15 minutes.

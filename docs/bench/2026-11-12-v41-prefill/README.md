# V4.1 prefill: direct census and exact dataflow experiments

Status: MID masked index promoted; seven rejected inference levers removed.
Wide PV fails reversed served root; short-index and cooperative experts are prepared, default off.
Rejected evidence remains. Branch `perf/v41-prefill`,
base `5ee5c8c4f`. M5 Ultra, unchanged GSQ-RCO IQ2_XS four-shard GGUF plus
disk IQ4_NL Engram, 384 experts/top-6. Both A/B arms use `iter`; serving must
use release. Runs are queued under task `v41-prefill`, holds at most 15 minutes.
Every load settles continuously for 60 seconds at pressure <=1; execution
permits 2 and aborts at 3. Shared ghost/compiler activity is retained in monitors.

Controller correction `ac45a0460`: the initial runner and child both added
`nice +10`, yielding effective niceness 19. Those captures below remain
**screens at niceness 19**, not final timing qualification. `d2d08d5d1` repeats
the census, isolation and native cells at verified niceness 10; child niceness
is now recorded in each monitor. Bitwise witnesses are unaffected by scheduling
priority. The release build used a single niceness adjustment throughout.

Raw evidence: `metagross:$HOME/qwen-runs/2026-11-12-v41-prefill/`.
The corpus is the existing uncycled code witness, 164,054 tokens under this
artifact's tokenizer. Timings exclude model loading and state restoration.

## Verified niceness-10 census and native qualification

All model children in this packet record effective niceness 10. Kernel entries
are **isolated GPU ms / matrix TFLOP/s / logical payload GB/s**. The root sample
is one 4096-row CED encoder chunk at 40960..45056 in a 49152-row root; append
is 1024 rows at 81920. Both complete captures have zero overflow and unchanged
endpoint bits. Payload definitions and instrumentation limits are stated below.

| Stage | Late-root chunk, 20 layers | Append, 40 layers |
|---|---:|---:|
| Routed gate/up | 309.36 / 74.97 / 178.3 | 276.09 / 42.00 / 366.0 |
| Routed down | 164.27 / 70.59 / 227.6 | 139.87 / 41.45 / 413.5 |
| Dense projections | 379.35 / 71.57 / 105.2 | 236.62 / 57.30 / 88.2 |
| Attention QK | 58.17 / 54.35 / 2002.9 | 27.74 / 59.45 / 2186.9 |
| Attention PV | 117.19 / 26.97 / 1356.1 | 61.70 / 26.73 / 1340.0 |
| Attention sinks | 49.56 / — / 975.7 | 24.71 / — / 978.6 |
| Inverse RoPE/layout | 77.83 / — / 275.9 | 27.86 / — / 385.3 |
| Query/index RoPE | 43.77 / — / 507.5 | 48.47 / — / 230.5 |
| Index scoring | 78.26 / 27.66 / 1744.8 | 162.94 / 27.59 / 1738.4 |
| Top-k | 7.36 / — / 153.8 | 21.84 / — / 103.9 |
| mHC | 121.32 / 1.33 / 502.2 | 51.22 / 1.57 / 592.9 |
| Engram GPU projection+gate | 58.30 / 44.20 / 59.0 | 13.78 / 46.75 / 74.1 |
| Other publication/pack/route/conversion | 341.95 / — / — | 133.14 / — / — |

Matrix rates as fractions of the supplied **131.7 TFLOP/s Half-tensor
reference**: root experts
56.9/53.6%, dense 54.3%, QK 41.3%, PV 20.5%, index 21.0%; append experts
31.9/31.5%, dense 43.5%, QK 45.1%, PV 20.3%, index 20.9%. The indexer uses F32 simdgroup matrices; mHC's F32 projection-only credit
also includes non-matrix work in the stage time. Neither has a measured
physical roof established by this Half-tensor reference.
By fractional distance from that reference, PV and index scoring lead. Absolute sampled excess
above the tensor compute floor ranks root **dense 173, GU 133, PV 93, down
76, index 62 ms**; append **GU 188, dense 134, index 129, down 96, PV 49 ms**.

Native baseline means after the excluded warmup pair:

| Cell | Wall ms | GPU ms | Between-command gaps ms | Host encode ms |
|---|---:|---:|---:|---:|
| Empty-state 48K CED root | 21256.35 | 17074.14 | 3771.00 | 1163.89 |
| 1K@80K Full append | 1069.13 | 1015.40 | 20.73 | 1.83 |

Root inclusive host preparation is 1211.63 ms, including 735.16 ms exposed
Engram preparation. Worker I/O counters are TLS-blind, so no worker read rate is
invented. The baseline scopes observe 11.27 MB / 0 mean process disk reads for
root/append after warming; these are whole-scope process I/O, not attributed
shader traffic. The separate resident-table 4K@40K read/decode probe emits
201.327 MB of complete F32 rows in 137.223 ms baseline (**1.467 GB/s decoded
output**); that is a CPU output-payload rate, not native-worker physical disk
bandwidth. Its full bits and per-layer bytes are retained in
`data/nice10-stage-engram-permute-summary.json`. CPU encoding/gaps have no
tensor FLOP or streaming-byte roof.
The sampled root chunk has 2074.52 ms GPU envelope and 1806.67 ms isolated cores;
the append cores sum to 1225.98 ms. Do not substitute these sampled sums for
the native command times above. Root whole instrumented wall is 24563.43 ms.

The rejected streamed lookahead passes native qualification here:
**21256.35 -> 19912.40 ms (1.06749x)**, three of three pairs positive, all
endpoint/state/following logits bitwise, pressure <=1; exposed preparation plus
encoding **1899.04 -> 460.48 ms (4.124x)**. Gaps fall 3771.00 -> 2474.82 ms
while GPU time stays 17074.14/17056.57 ms. The inactive append control is
1069.13/1073.94 ms (-0.45%). It is rejected by the served gate below.
Data: `data/nice10-{root,append,native-profile,cell-prefetch}-summary.json`.

Remaining isolation gates at verified 10: masked scoring, three retained pairs,
**162.983 -> 107.521 ms, 1.51582x**, exact full causal/output witnesses.
Normalize/RoPE N512 is **389.639 -> 311.102 us, 1.25245x**, exact; the gain is
close to the isolation bar and still needs native cells and serving.

## Initial niceness-19 native census

Initial probe source `ac2be3229`; packet controller `3a4ad5ace`.

| Cell | Wall ms | GPU command ms | Other observations |
|---|---:|---:|---|
| 49,152-row streaming CED root | 28,009.48 | 17,025.76 | 12 x 4096 encoder rows; pressure 0 |
| 1024-row Full append at 81,920 | 1,816.87 | 1,034.94 | 3 commands; pressure 0 |

The append is below the 2541-row CED dependency halo. Its Engram source reads
49,152 hash rows / 7,077,888 logical bytes in 48.35 ms; advice costs another
204.53 ms. Process disk reads are 264,830,976 bytes. These are different traffic
domains. The two exposed Engram wait spans total 262.46 ms. Driver scheduling
intervals overlap command/wait and must not be added as independent wall costs.

The root has 4,047.03 ms of encoder Engram preparation and 4,531.76 ms of
inclusive host preparation. Worker-thread Engram reads are absent from the
TLS-scoped source counters, so their zero counters do not mean zero I/O.
Process disk reads are 13,106,413,568 bytes. The gap between the 12 encoder
commands and the encoder's host encode spans remain separate attribution terms.

Root host encode is 4482.22 ms. The 22 command intervals have **9288.73 ms**
of uncovered GPU-clock gaps, of which 9142.56 ms separates encoder chunks.
The append has 746.85 ms between its three command intervals. Native GPU duty
is about 60.8% / 57.0% of root/append wall. Inclusive driver scheduling is not
added again. Wall outside the first/last command envelope is 1694.98 / 35.08 ms.

The first append dispatch profile has 2816 samples, no overflow, identical
endpoint logits, and wall 1508.44 ms. Profiling splits encoders, removes
concurrency and changes scheduling; use its kernel ranking, not its wall as an
A/B improvement. Its routed GU/down cores total 287.43/144.00 ms; dense matrix
cores 243.90 ms, index scores 163.02 ms, QK 28.08 ms, PV 61.98 ms. Useful routed
rates are 40.35/40.27 TFLOP/s (30.6% of 131.7), dense 58.23 TFLOP/s (44.2%).
Attention/index figures are provisional until constant-buffer shape accounting.
Logical operand GB/s is not measured DRAM bandwidth; reuse can exceed the
1175.47 GB/s measured memory reference.

The initial whole-root dispatch attempt overflows at 65,536 samples (216,618
unrecorded dispatches). It is rejected for stage accounting. `3a5d4de57` samples
only one 4096-row encoder chunk at 40,960 instead, before reconstruction.
`93bcc5aa9` additionally records constant-buffer words for exact key counts,
row offsets and projection work. Native timings above are unaffected by this
instrumentation correction.

## Historical niceness-19 kernel census

`census2-root` samples the 4096-row encoder chunk at 40960..45056 in a 49152
root: **51,405 dispatches, zero overflow**, unchanged complete endpoint bits.
The new append capture is the baseline arm of `stage-index`, also complete and
bitwise. These tables are isolated sampled kernel cores, not additive native
wall attribution. The sampled root chunk's command GPU envelope is 2265.40 ms,
versus 1449.96 ms in the earlier native run; its kernel cores sum to 1779.23 ms.
Append sampled cores sum to 1245.93 ms versus 1034.94 ms native commands. Encoder
splits, instrumentation, residency and shared machine drift perturb the samples.
The instrumented root's 49.16-second whole wall is not a native speed result.

Each entry is **GPU ms / matrix TFLOP/s / logical GB/s**. Non-matrix stages get
no fabricated tensor FLOP rate. mHC credits only its F32 mixing projections.
All rates use constant-buffer work dimensions, including each query's actual
visible key count. Engram's GPU row includes its projection plus gate; the dense
row excludes it. Sinks include softmax and normalization; inverse RoPE is separate.

| Stage | 4K late-root encoder chunk, 20 layers | 1K append at 80K, 40 layers |
|---|---:|---:|
| Routed gate/up | 310.23 / 74.76 / 177.8 | 297.47 / 38.98 / 339.7 |
| Routed down | 152.57 / 76.01 / 245.1 | 148.96 / 38.93 / 388.3 |
| Dense projections | 382.39 / 71.00 / 104.4 | 228.99 / 59.21 / 91.1 |
| Attention QK | 57.04 / 55.42 / 2042.6 | 28.39 / 58.09 / 2136.6 |
| Attention PV | 116.72 / 27.08 / 1361.4 | 63.07 / 26.15 / 1310.9 |
| Attention sinks | 49.95 / — / 968.2 | 24.68 / — / 979.8 |
| Attention inverse RoPE/layout | 78.43 / — / 273.8 | 28.22 / — / 380.5 |
| Query/index RoPE | 42.37 / — / 524.2 | 49.62 / — / 225.2 |
| Index scoring | 78.58 / 27.55 / 1737.6 | 163.25 / 27.53 / 1735.1 |
| Index top-k | 5.23 / — / 216.5 | 20.69 / — / 109.7 |
| mHC | 110.78 / 1.45 / 550.0 | 52.10 / 1.55 / 583.0 |
| Engram GPU | 52.51 / 49.08 / 65.5 | 20.23 / 31.85 / 50.5 |
| Other publication, pack, routing and conversions | 342.44 / — / — | 120.27 / — / — |

**Traffic interpretation:** logical operand payloads can be served from cache,
so >1175 GB/s does not violate the measured DRAM roof. Experts use one complete
packed-bank pass plus activation/output payload as a reference; actual expert
occupancy, repeated panels and physical DRAM traffic are not inferred. Top-k
uses a single score scan plus output payload, not all radix passes. These are
effective payload rates, not bandwidth counter measurements. Host encoding and
command gaps perform no tensor work; they have no tensor TFLOP/s or data-movement
GB/s roof. Disk Engram source rates/bytes are reported separately above.

Against **131.7 TFLOP/s**, root/append fractions are: experts **56.8–57.7% /
29.6%**, dense **53.9% / 45.0%**, QK **42.1% / 44.1%**, PV **20.6% / 19.9%**,
index **20.9% / 20.9%**, Engram projection+gate **37.3% / 24.2%**. mHC's F32
projection-only rate is 1.1–1.2% of that Half-tensor reference; it is not an
eligible Half tensor matmul, so that comparison is not its physical roof.

Against the nominal tensor reference, **PV and index scoring** lead, then append
experts, then QK/dense/root experts. By absolute sampled ms above the tensor
compute floor, the root leaders are **dense 176 ms, GU 134 ms, PV 93 ms**;
append leaders are **GU 209 ms, index 129 ms, dense 126 ms, down 105 ms**.
These floors omit mandatory unpack, gathers, scalar/SFU work and memory effects.
The root's 9.29 seconds of native command gaps remains the larger end-to-end
opportunity. The 342 ms root residual also exposes the per-row publication path:
10,240 compress-publish, 10,240 index-key GEMV, 10,240 RMS-norm, 12,288 copy and
6144 pool dispatches in one 4K chunk. Extending qualified bulk publication to
4K long-depth rows is a distinct next lever if needed.

Reproducer: `scripts/profile/v41_prefill_census.py RAW.jsonl --output SUMMARY.json`.
Compact grouped/per-kernel evidence is in `data/root-summary.json` and
`data/append-summary.json` beside this note.

## Source comparison and licenses

Inspected only the four requested repositories, using `rg`. Local revisions:
mlx-serve `da964cf12a899772285be58e4740de6f75c9f6af`, MLX
`93007e9a8a44032ebc9d21d37b7dee14b7f38f1b`, deepseek-v41-mlx
`8bdd543c7160800f3451d9595c996129b7abecdf`.

* qwen uses 4096 input rows: mean 64 routed rows/expert; a 1024 append has mean
  16. Its existing sorted native-IQ experts use 64 routed rows/K64, decode into
  shared Half slabs, and reuse that slab across up to two 32-row products.
  Both GU and down already skip products outside the live row span; a 16-row
  mean must not be described as unconditional fourfold wasted 64-row math.
  The first live product still has a 32-row descriptor. GU fuses clipped
  activation, weighting and Half storage, preserving F32 boundaries.
  The established shader already retains the mlx-serve MIT attribution.
* mlx-serve's V4.1 `MOE_ROWS` is 8192: mean 128 routed rows/expert. Its NAX
  gather planner uses 32/64/96/128 row tiles and K64/K128, including a paired
  GU path; the 128-row paired rung starts at mean 96 rows/expert. This affine/
  FP4 planner is distinct from its native-GGUF implementation. Native GGUF
  tile code dequantizes into cooperative tiles. No requantization is proposed.
* MLX `GatherQMM::eval_gpu` selects the sorted RHS path when M=1, B>=16 and
  B/E>=4. Its NAX kernels use K64 and 32/64 row rungs; quantized metadata
  remains attached to the tile loads. Steel NAX attention at D512 uses BQ32,
  BK32, WM2/WN4 and sink-aware online softmax. This dense SDPA layout is not
  directly substitutable for per-query sparse MLA gathers.
* mlx-serve sparse QK uses 64-key/K64 panels with double-buffered shared keys;
  PV uses 32-key/D128 panels and independent output-dimension groups. Its
  grouped PV epilogue also performs inverse RoPE/layout conversion.
* deepseek-v41-mlx uses SwitchGLU, float32 score einsums and materialized
  gathers in sparse attention, processed in 256-query spans; candidate masks
  precede argpartition and chronological sorting. Moving route weights after
  down is algebraically linear but does not preserve our rounding contract.

MLX is MIT (Apple, 2023). mlx-serve's own code is MIT (David Dalcu, 2026),
with third-party Apache-2.0 terms on the gather NAX port. deepseek-v41-mlx is
Apache-2.0. Licenses and per-file headers were checked before implementation.
These new prototypes use qwen's existing formulas/dataflow; no additional
external shader body was copied.

## Lever 1: normalization and inverse RoPE — rejected at native wall

Prototype `0b3893400`; `QWEN_V41_ATTN_NORMALIZE_ROPE`, default off. Combine
two existing attention output passes, retaining a volatile F32 quotient boundary
under fast math. No new backing allocation or chunk/operand change.

An M4 Max development sanity run in `iter` compares every output bit at 1,
127 and 512 rows, including invalid denominators. All 20,971,520 elements
match. Three ABBA blocks give 1.672/1.609/1.901x isolated rates. This is not
the M5 gate. The initial M5 niceness-19 screen is also bitwise: N512
345.858 -> 257.295 us, 1.344x. Verified-niceness-10 repetition is **389.639 -> 311.102 us, 1.25245x**
at N512, with every output bit equal. Native cells and serving remain pending.

### Native decision, 2026-11-13

All eight arms of both cells are bitwise, at niceness 10 and pressure 0.
Retained 48K root pairs (baseline/candidate ms) are
31567.651/42394.118, 21301.646/20644.604 and 20368.453/20232.547.
Means **24412.583/27757.090 ms (0.87951x)**, 2/3 positive. Physical disk
read variation is large: early warmed arms read 7.6–15.7 GB while GPU time
stays near 17 seconds. Keep every declared retained pair; do not trim the
slow root. This packet does not qualify a root wall win.

The 1K@80K append is **1069.475 -> 1061.931 ms (1.00710x)**, 3/3 positive.
Reject the broad candidate at the root gate, remove the shader/knob/price
example and move to PV accumulator writeback. No serve promotion.
`data/nice10-cell-normalize-summary.json` retains both cells and all pairs.

## Lever 2: masked index scoring at agent depths

Prototype `95d622e9a`; `QWEN_V41_MASKED_REINDEX_MID`, now default on
after the gates below; `=0` restores the 128K eligibility threshold. Enable the
existing exact mask-skipping F32 MMA kernel for ordinary 512..2048-row appends
from 49,152 to 131,071. The existing mask rollback still vetoes it. Allowed
scores retain their MMA and head-sum order; rejected scores remain excluded by
selection. No new memory, precision, candidate count or selection policy.

The existing 128K guard leaves the requested 80K append on full-domain scoring.
Measure all score dispatches on the actual append, require >=1.25x in the
isolated dispatch census and bitwise logits/causal state/following decode,
then at least three native chronological pairs, then three-agent serving.

Initial niceness-19 M5 screen: **163.250 -> 108.310 ms, 1.507x** over all
index-score dispatches. Complete endpoint, causal state and following decode
are bitwise. Verified-niceness-10 repetition is **162.983 -> 107.521 ms, 1.51582x**,
three retained pairs with the same complete bitwise witnesses. Native/serve
gates are recorded below.

### Native MID gate, completed 2026-11-13

Source `f69e9edeb`, effective niceness 10, same `iter` binary and restored
prefix in every arm. The predeclared warmup pair is excluded; retained order
is BA/AB/BA. At 1K@80K, baseline/candidate pairs in ms are
1053.505/1005.649, 1061.304/999.578 and 1054.623/999.916.
Means **1056.477 -> 1001.714 ms, 1.05467x**, 3/3 positive, pressure 0.
All eight arms' endpoint/state/following-teacher-step witnesses are bitwise.
Inactive 48K root means are 20048.214/20121.080 ms (-0.36%); claim no root gain.
The first baseline root takes 45043 ms and stays in raw evidence.
Compact evidence: `data/nice10-cell-masked-summary.json`.

### First MID served block, AB (reverse block pending)

Release SHA256 `08725bb811ecc9497c9c383be2d040ec7cbbf19bc180a34a55530740e76f18a6`,
niceness 10 recorded for both model children, cap 32, pressure 0/0. All nine
eligible equal-size appends improve; mean ratio **1.03881x**, median **1.03663x**.
All matched short appends: mean **1.03200x**, median **1.02924x**. Delivered
throughput **78.575 -> 80.517 tok/s**, gap p99 **0.15200/0.15126 s**.

Roots are inactive controls: native A **37.336/55.510/26.276 s**, B
**27.108/22.152/21.293 s**. Keep this large variation; claim no root speedup.
Visible client TTFTs are A 37.45/55.61/null, B 27.21/22.23/null seconds.
All three native root phases remain represented. No promotion before the
predeclared reversed block. `data/serve-masked-pair1w32-comparison.json`.

### MID reversed block and promotion, 2026-11-13

Same release SHA and cap 32. Reversed BA: all nine eligible matched appends
improve, mean **1.09442x**, median **1.07055x**. All matched short appends
mean **1.05123x**. Delivery **78.458 -> 80.283 tok/s**, gap p99
**0.15402/0.15110 s**. NI10 verified. Baseline A reaches pressure 2 for part
of the steady window; candidate B stays at 0. Pressure 2 is allowed and kept,
and this limits attribution of the larger reverse-block magnitude.

Across the two order directions, all 18 eligible matches improve. Visible
eligible client TTFT means are **1.5692 -> 1.5047 s (1.04283x)** in AB
and **1.6076 -> 1.4795 s (1.08655x)** in BA; keep every queued outlier. The separate native 80K cell establishes **1.05467x** over
three chronological pairs, and the isolated stage **1.51582x**, all bitwise.
**Promote MID default on with `QWEN_V41_MASKED_REINDEX_MID=0` rollback.**
The existing `QWEN_V41_MASKED_REINDEX=0` veto remains.

The reverse roots are inactive controls, A **21.238/20.590/20.623 s**, B
**21.116/23.973/24.120 s**; visible TTFTs A 21.33/20.67/null, B
21.20/24.04/null. Keep all phases, claim no root speedup. All three sessions
fit and decode reaches width 3. `data/serve-masked-pair2w32-comparison.json`.

## Lever 3: streamed CED Engram lookahead — rejected at serving

Prototype `a75e4bce1`; former `QWEN_V41_CED_STREAM_PREFETCH`. Code and knob
removed after the served gate; both CED source files are restored to the base.
Streaming
previously passes neither prefetched rows nor lookahead hashes to the existing
encoder prefetch machinery. Hash/decode the next chunk while the current GPU
command runs, join the scoped worker before yielding, and retain only decoded
CPU values plus their exact token/start identity. Quantum or token changes
discard the cache. Logical row partition, causal transitions and arithmetic stay.

Enable only when the existing pipeline/prefetch CPU allowance is admitted.
Decoded next values + hash budget + tokens + 64 KiB metadata must fit that
second upload envelope. No extra GPU backing or live worker/command at yield.
The isolated gate prices exposed Engram preparation on an 8192-row episode;
then compare complete 48K roots with at least three interleaved native pairs.
Numerical gate is full causal-state digest plus complete endpoint and following
decode logits, compared bitwise from the same snapshot. Serving remains required.

Niceness-19 screen, **not final qualification**: after the predeclared first
warmup pair, the three root pairs are 20338.85/19315.64,
20393.02/19673.60 and 20669.59/19816.96 ms (baseline/candidate). Means
**20467.16 -> 19602.07 ms, 1.04413x**, all three positive, maximum pressure 2.
Retained exposed Engram preparation + encoder spans improve **3.614x**.
The inactive 1K@80K control is 1076.61 -> 1070.36 ms; claim no append win.
Every arm's endpoint, causal-state digest and following decode are bitwise.
Both warmup arms remain recorded: the first root's 42.45 -> 19.37 seconds
shows why cache order cannot qualify the large cold-looking ratio. The initial
8K 918.30 -> 58.66 ms preparation screen has the same order caveat.
Compact evidence: `data/cell-prefetch-summary.json`.

### Served decision, completed 2026-11-13

Same release binary in both arms, SHA256
`234dbfda6c632328ed3fc48d18474d4105f1810c21cf7bd95433fa31b667c3d6`,
three 49152-token source roots, context ceiling 196608, 120-second steady window,
seed 20261112, actual agent sampling, niceness 10. Both pre-load and post-warm
settling require 60 continuous seconds <=1; execution permits 2 and aborts at 3.
All three sessions fit and decode reaches width 3.

| Block | Root native prefill baseline/candidate ms | Equal-size short append mean ratio | Delivered tok/s baseline/candidate | Pressure baseline/candidate |
|---|---:|---:|---:|---:|
| AB, warm reply cap 8 | 26912.36 / 20821.28 (1.2925x) | 1.0120x, 12 matches | 66.31 / 68.22 | 0 / 0 |
| BA, warm reply cap 32 | 21717.81 / 22252.85 (0.9760x) | 0.9623x, 14 matches | 82.88 / 72.89 | 0 / 2 |

The third warm reply has no visible delta at either cap; report its native
prefill phase and leave client TTFT null. All three native root phases are
retained. The other client root TTFTs are AB 25.73/27.16 -> 21.17/20.88 s;
BA baseline 22.05/21.68 versus candidate 23.56/21.88 s. The blocks use different
warm reply budgets and are not pooled into a paired confidence interval.
Short appends match label and exact new-row count, require offered-in-window
and completed in both arms, exclude carry-in/ramp, and retain all matches.

The reverse block does not retain a root or append benefit. Pressure 2 is
allowed and is not an exclusion. **Do not promote or repeat this unchanged in
search of a favorable result.** Keep the exactness/native evidence and the
negative serve record; move to index scoring. The code adds no new GPU backing,
but its native latency win does not establish a durable served win in this
packet. Compact served reductions are `data/serve-prefetch-pair*-comparison.json`.

## Lever 4: PV accumulator writeback fusion — rejected at serving

`QWEN_V41_ATTN_PV_ROPE`, default off. On the existing H64/D512 tensor64
path, retain the K32/D128 matmul sequence and 12 KiB shared operands. Normalize
F32 accumulator pairs, retain volatile F32 quotients, inverse-rotate the final
64 coordinates, then write directly into query-major output. Remove the PV
slab write plus normalization and layout passes. No new backing or enlarged
threadgroup allocation. Runtime coordinate checks reject an unsupported pair
layout through the existing status word.

The cooperative-pair epilogue is adapted from mlx-serve's MIT
`src/plugins/deepseek_v41/kernels/attn_pv.metal` at the inspected revision.
The shader retains its full MIT notice. qwen's existing trig, normalization
and status behavior replace mlx-serve's formulas. Do not alter the PV grid:
the earlier dimension-split campaign is a retained negative, not this lever.

First gate: 1K@80K dispatch capture, three retained interleaved pairs after a
warmup pair, complete endpoint/causal/following-step bits equal, and >=1.25x
for the sum of PV + normalization + inverse RoPE. No native or served claim
until this gate and subsequent real-cell/serve gates pass.

### PV isolated gate, completed 2026-11-13

M5 source `db0ef909f`, same `iter` process, effective niceness 10. Retained
combined PV/normalization/inverse-RoPE pairs in ms are
104.534/68.436, 104.689/67.279 and 104.778/66.680.
Means **104.667 -> 67.465 ms, 1.55143x**, 3/3 positive, pressure 0.
All eight arms' endpoint logits, complete causal state and following teacher
step are bitwise. Warmup pair remains in raw evidence; no native wall claim
from dispatch instrumentation. `data/nice10-stage-pv-rope-summary.json`.

Native root and append qualification keeps `QWEN_V41_MASKED_REINDEX_MID=1`
in both arms, testing the prospective combined configuration. MID's reverse
serve gate is still pending; this does not promote either candidate. Other
experimental policies stay off. Regate if the final baseline changes.

### PV native gate, completed 2026-11-13

MID=1 in both arms, PV 0/1, Engram ordering off. Same `iter` binary from
`db0ef909f`, NI10. The three retained 48K root pairs are
21167.689/20788.287, 19942.193/19403.946 and 20102.654/19685.690 ms:
**20404.178 -> 19959.307 ms, 1.02229x**, 3/3 positive, pressure <=1.
The append pairs are 1016.521/989.486, 1009.953/988.544 and
1015.828/997.813 ms: **1014.101 -> 991.947 ms, 1.02233x**,
3/3 positive, pressure 0. All eight arms/cell have identical complete
endpoint/state/following-step signatures. Both baseline signatures also equal
the earlier original-production reference, including the 80K prefix now
constructed with MID on. `data/nice10-cell-pv-rope-summary.json`.

Serving qualification is two opposite-order blocks, same release binary,
cap32, 120-second windows, with MID=1 and ENGRAM_SORTED_ROWS=0 fixed in both
arms. Report every root phase, visible root TTFT and every matched short
append. PV is still default off until that gate passes.

### Broad PV served decision and late-append scope

Same release SHA `fd7206b74a2008ccb636408d62ab272c4a88a193fea2282d73432656472b2dbf`,
MID=1/Engram ordering=0, cap32, NI10. AB root means **25.338 -> 24.266 s
(1.04421x)**; reversed BA **24.160 -> 28.873 s (0.83679x)**. AB candidate
reaches allowed pressure 2, retained; both BA arms stay at 0. **Reject broad
root eligibility; no default promotion on roots.**

Matched ordinary short appends improve in both blocks: AB mean **1.00757x**,
15/20 positive; BA **1.03169x**, 18/20. Visible append TTFT ratios are
**1.03448/1.03353x**. Delivery AB **79.483/78.775**, BA **81.300/81.508**
tok/s. All three root phases and null visible third-root TTFTs are retained
in `data/serve-pv-pair{1,2}w32-comparison.json`.

Keep only the already-measured late ordinary append geometry for additional
eligibility validation: M5 Ultra, H64/D512/R64, 128..2048 logical rows,
start >=49152, no active CED context. Root/short-comparison cells retain the
existing path. A probe-only call counter must show zero fused calls there,
and positive fused calls at 1K@80K; all paired bitwise assertions remain.
This scope is not promoted yet. It changes no arithmetic of the matched
append cohort. Broad pp4096/8K gains of **1.02734/1.02411x** are retained as
rejected-scope observations, not default throughput gains.

### Guarded PV native validation

Source `64036e4a4`, MID=1/PUBLICATION_4K=0 fixed, same `iter`, NI10,
pressure 0. Every one of eight arms per cell passes the unchanged complete
bitwise assertion. Root/pp4096/8K fused-call counts are **0 in every arm**.
At 1K@80K, baseline calls are 0 and candidate calls are 80 per arm.

The retained append pairs are 993.963/974.587, 995.222/975.261 and
994.999/975.763 ms: **994.728 -> 975.203 ms, 1.02002x**, 3/3 positive.
Inactive controls, baseline/candidate means: 48K **24083.399/24061.419 ms**;
pp4096 **2712.746/2743.014 ms**; 8K **3609.864/3672.779 ms**. Keep their
ambient variation; claim no root/short-prompt throughput change from an
optimization that makes zero calls there. Opposite-order B3 packets with the
new guard and fresh release are queued before any promotion.
`data/nice10-cell-pv-narrow-summary.json` retains all four cells.

### Guarded PV first served block

Fresh release SHA `c27837e7c15acfae21b4b4690e263bfe49eccd1ea15db4ad3102a0c25dcace14`,
MID=1/PUBLICATION_4K=0, cap32, NI10, pressure0 in both arms. AB matches
21 ordinary short appends with equal actual row counts: **1.02596x** mean
prefill ratio, **21/21 positive**, median pair 1.02263x. Visible append
TTFT ratio is **0.99097x**, retained without reclassification. Delivered
throughput is **80.975/82.058 tok/s** and gap p99 **0.15279/0.14981 s**.
Inactive root mean ratio **1.00063x** is a control, not a PV root gain.
The opposite order must finish before promotion.
`data/serve-pv-narrow-pair1w32-comparison.json` retains every matched call.

### Guarded PV reversed served gate — rejected

Same fresh release SHA and cap32; both arms remain pressure0 and NI10.
Reversed BA has **15/15** matched short appends faster in prefill,
mean **1.01971x**, median **1.01667x**. However complete client TTFT mean
ratio is **0.92612x** (12/15 positive, median 1.01880x), delivery is
**87.250/81.933 tok/s**, and gap p99 **0.05255/0.15198 s**.
The inactive root ratio is 1.00095x. Keep every match and the long TTFT
calls; do not select the favorable median or isolated prefill metric.

**Reject at realistic serve.** Remove the PV kernel, product flag, scoped
experiment and call counter; restore the tensor shader exactly to base.
No unchanged rerun seeking promotion. AB and BA together have 36/36 positive
prefill pairs, but that does not overcome the served TTFT/delivery regression.
`data/serve-pv-narrow-pair{1,2}w32-comparison.json` retains both orders.

The already-queued publication wall packet uses MID=1/PV=1 in both arms.
Publication's root and 4096-row Full cells make zero guarded PV calls;
those root witnesses remain independent. Its inactive 1K append control
will be repeated with PV removed before a publication promotion. Served
publication qualification must use MID on and PV off.

## Lever 5: in-place Engram permutation — rejected at native wall

`QWEN_V41_ENGRAM_SORTED_ROWS`, default off. Reuse the temporary chunk ID
vector for `(table row, original position)` tags; sort it in place, read/decode
each distinct row once and scatter its exact F32 values into original token
order. Repeated rows copy their first decoded result. Source-stamp validation,
row bounds, codecs and adaptive read-ahead stay. Advice visits unique sorted
rows. Decode's single-token multi-layer path is unchanged.

The permutation uses 20 position bits, bounds the input to 1,048,576 IDs and
checks row-ID packability before mutation; larger/unpackable requests retain
the original reader. Production chunks have at most 4096 token rows. The output
allocation is the existing full decoded chunk, and sorting uses the existing
ID vector; the per-block output and offset temporaries are removed. No table
cache, larger GPU backing or increased CPU high-water requirement.

Stage price `examples/v41_engram_price.rs`: actual uncycled 4K chunk hashes at
40960, after a real CED prefix loads the resident model. Three retained
interleaved pairs after one warmup pair. Time CPU read/decode only; address
generation, distinct-row counts and complete F32-output digests are outside
the timer. Require
>=1.25x and bitwise values, then native paired 48K/80K cells and B3 serving.
The existing CPU fixture now also checks IQ4_NL, repeated/reordered rows across
probe and block boundaries, exact scatter values and a late invalid row.
No new ignored test. Stage and model gates remain pending.

### Engram isolated gate, completed 2026-11-13

Source `db0ef909f`, `iter`, recorded niceness 10, pressure 0. Retained
baseline/candidate read-decode ms: 142.051/115.354, 135.623/107.737 and
133.995/105.654. Mean **137.223 -> 109.581 ms, 1.25225x**; all three pairs
positive, the first pair is 1.231x. This narrowly clears the mean stage bar.
Each arm hashes all **50,331,648 F32 values**, 201,326,592 output bytes,
with identical complete per-layer digests.

The 4K chunk requests 98,304 rows per layer; distinct counts are 42,418/42,420
for layers 1/14. Unique-row reading therefore removes about 56.85% of row
reads/dequantizations. This is not a claim about physical disk-byte reduction.
`data/nice10-stage-engram-permute-summary.json` retains all three pairs.
The feature-enabled CPU control/candidate/rollback scope fixture also passes.

Native qualification holds MID=1 and PV_ROPE=1 in both arms, testing the
prospective next baseline. These candidates still need their own wall/serve
gates; if either is rejected, regate Engram with the resulting baseline.
No default promotion follows from this marginal isolated result.

### Engram native decision, completed 2026-11-13

MID=1/PV_ROPE=1 fixed in both arms, NI10, pressure <=1. Three retained 48K
root pairs baseline/candidate ms: 20245.159/20029.784,
20323.944/20548.062, 20760.000/21211.200. Means
**20443.035/20596.349 ms (0.99256x)**, only 1/3 positive. Exposed Engram
preparation plus encode improves 1689.19 -> 1527.13 ms (1.106x), but wall
fails. Append **1004.513/1003.584 ms (1.00093x)**, 2/3 positive; claim no
append win. Every arm's complete output/state/following-step signature is
bitwise. Both baseline signatures equal the prior PV candidate, additionally
validating a globally enabled PV path through the full 80K CED prefix.

**Reject, remove reader, policy scope, knob and price example.** Restore both
Engram source/fixture files to base. No served promotion or unchanged rerun.
The old release source may retain the unused prototype with its flag fixed at
0 during PV A/B; the final source removes it. All active PV arithmetic is
unchanged. `data/nice10-cell-engram-permute-summary.json` retains both cells.

## Served MID cohort, declared before measurement

Retain every matched short append under the earlier rules, and additionally
report the eligible subset: both actual native row counts 512..2048 and both
reused depths 49152..131071. The subset follows the code guard; it does not
filter on timings. Root phases are inactive controls. Use warm reply cap 32
consistently for both order directions. Keep null visible client TTFTs.

## Lever 6: exact late-4K publication fusion — rejected at serving

`QWEN_V41_PUBLICATION_4K`, default off. The late-root sample emits 51,405
dispatches. Extend complete-group publication only to pruned ordinary 4096-row
batches on M5 Ultra at 32K+, retaining all earlier bulk paths and existing
COMPRESS_ROWS/COMPRESS_LONG_ROWS rollbacks. BF16_TG=0 vetoes the new route;
unaligned key weights retain the per-row path.

The existing bulk kernel changes the latent RMS tree and uses a different
BF16 projection tree. Its exact variant emulates the legacy >=512-thread RMS
with two independent lane partials and the same final simdgroup reduction.
Key projection repeats the legacy 128-thread dot8/simdgroup tree at K512,
including its two zero partials. Key norm, pooling and RoPE retain their
existing per-row expressions and storage boundaries. Scratch grows by about
1 KiB within the workgroup; no new model/session backing. These are adaptations
of the project's existing MIT-licensed kernels, not an external shader port.

Require >=1.25x isolated publication chain and complete bitwise output/causal/
following-step signatures over three retained chronological pairs. Do not
relax a failed exactness assertion. Only then run unsampled root/append walls
and realistic B3. Root dispatch samples now retain their true repetition ID.

### First exact-publication witness and state correction

The first model packet at `64036e4a4` stops after one warmup pair, before
any retained speed qualification. Complete endpoint and following-step logits
are bitwise equal, but the snapshot digest differs. Keep the failed packet
`stage-publication4k-ni10-*`; do not count its timing ratio as a win.

Snapshots retain pending value/gate bytes at an even ratio-two boundary. The
new bulk path must copy the last first-half input rows into those existing
buffers, matching the old chain. Odd head/tail batches retain the per-row
route. The corrected packet keeps the same whole-payload digest assertion;
there is no canonicalization, tolerance or comparison exclusion.

### Corrected exact-publication stage gate

Source `4298369aa`, same `iter`, recorded NI10, pressure 0. All eight arms
match unchanged full endpoint, entire stored snapshot payload and following
teacher-step hashes. The first failed witness remains separate.

Three retained isolated publication-chain pairs are 166.706/0.337,
167.725/0.332 and 166.650/0.332 ms. Mean **167.027 -> 0.334 ms**.
The reduction reflects replacing thousands of tiny dispatches with complete
batch launches. It is a per-dispatch counter measurement, **not a 500x engine
or native-wall claim**. The actual kernel launch has 2048 groups x256 threads,
ratio2, group0=20480 at the sampled 40K cut; one launch measures 73.833 us.
Candidate work was not omitted by the reducer. The candidate includes its
pending-byte copies. `data/nice10-stage-publication4k-summary.json`.

Next native wall packet holds MID=1/PV_ROPE=1 fixed, publication 0/1,
three retained pairs each at 48K root and 1K@80K. Guarded PV is inactive on
this root and active equally on both append arms. No publication promotion
before native and B3 gates.

### Publication prospective native cells and Full-depth witness

Source `4298369aa`, same `iter`, NI10, pressure0; MID=1/PV=1 fixed.
All eight arms in each cell pass complete endpoint/state/following-step bits.
Three retained pairs per cell:

| Cell | Baseline ms | Candidate ms | Ratio | Positive pairs |
|---|---:|---:|---:|---:|
| 48K CED root | 24131.437 | 23558.928 | 1.02430x | 3/3 |
| 1K@80K append, inactive publication control | 977.684 | 975.388 | 1.00235x | 3/3 |
| 4K@40K Full, all 40 layers | 3221.161 | 3052.374 | 1.05530x | 3/3 |

The Full cell exercises all layers beyond the sampled 20-layer CED encoder.
Root and 4K shapes make zero guarded PV calls; their exactness and timing
are independent of PV's failed serve gate. Keep the old prospective append
control as such. Rebuild from PV-removed source, repeat final-baseline native
root/append pairs, then run opposite-order B3 only if the wall gate passes.
No publication default change yet.
`data/nice10-cell-publication4k-prospective-summary.json`.

### Publication final-baseline native wall gate

Fresh source `4a040db01` has PV removed, MID on and publication off by
default. Both `iter` arms use SHA
`592913f9b40bc2b3cf5fc27241934934a6d53cf200306979a93fff8b38956f35`.
Three retained interleaved pairs after the declared warmup pair:
48K root **24267.477 -> 23671.029 ms, 1.02520x**, 3/3 positive.
The 1K@80K inactive control is **998.521/997.444 ms, 1.00108x**, 2/3 positive;
claim no append gain. All eight complete signatures per cell are bitwise,
NI10 and pressure0. `data/nice10-cell-publication4k-final-summary.json`.

B3 AB/BA now uses release SHA
`a8c9eb0d727cc2c4ff3c52b246cc9a02584dadde389bff36fc3a132981e1ea7c`,
MID=1 and PUBLICATION_4K=0/1, cap32, identical plan and thermal protocol.
Root prefill/client TTFT are the active target; short appends are controls.
No default change before those served gates.

### Publication first served block

Same fresh release SHA above, NI10/pressure0, cap32. AB keeps every root:
baseline **25014.205/24939.212/24558.496 ms**, candidate
**23919.958/23877.341/23505.504 ms**. Means **24837.304/23767.601 ms,
1.04501x**; all three pairs improve. Visible first/second root TTFTs are
**25.098/25.011 -> 24.004/23.947 s**, ratio **1.04501x**; third remains null.
Short append controls: 15 matches, mean prefill **1.00316x**, client TTFT
**0.99173x**, delivery **85.483/80.767 tok/s**, gap p99
**0.05346/0.15221 s**. Preserve these unfavorable controls alongside the
active root benefit; no aggregate delivery claim. BA is required.
`data/serve-publication4k-pair1w32-comparison.json`.

The first automation completed AB, then stopped on its reducer's flat-path
assumption. The raw artifacts are complete; reduce them without rerunning
models and launch the predeclared BA block. The reducer now accepts both
native nested directories and flattened copies, verified on these exact
artifacts with identical reductions.

### Publication reversed served gate — rejected

Same release binary, cap32, NI10 and pressure0. BA retains all three roots:
baseline **20744.903/20308.901/20385.663 ms**, candidate
**23797.047/23264.774/22778.469 ms**. Means **20479.822/23280.097 ms,
0.87971x**; all three pairs regress. Visible TTFT ratio **0.87278x**;
third remains null. Short append controls: 15 matches, prefill **1.00230x**,
client TTFT **1.01598x**, delivery **81.000/87.192 tok/s**, gap p99
**0.15074/0.05254 s**. Baseline load-after reaches 8.66; keep it and every
pressure0 sample. No causal attribution or exclusion on shared CPU load.

**Reject at realistic root serve.** The exact stage and two native root gates
do not overcome this reversed result. Remove the publication shader, product
flag, scoped selector and test-only prototype changes; restore publication,
BF16 visibility and direct-call fixture files exactly to base. Keep all
snapshots/digests and native/served evidence. No unchanged rerun seeking a win.
`data/serve-publication4k-pair{1,2}w32-comparison.json`.

## Lever 7: adaptive hot-expert panels — rejected at isolation

The 8K review finds a numerical boundary beyond memory: at a nonzero reused
prefix, a wider batch can cross the existing pruned-publication threshold
and select a different latent RMS tree. This needs explicit lineage or the
paired non-inferiority protocol; do not silently relax the bitwise digest.
First isolate the expert-reuse component at the admitted 4K geometry.

Default-off `QWEN_V41_EXPERT_PANELS_128` applies only to 4096-row CED encoder
workspaces on the existing native-IQ tensor tier. Runs with >=96 rows use
M128 panels; smaller runs retain M64 panels. A GPU plan builds two disjoint
panel prefixes over the unchanged sorted row offsets. The extra E+1 integers
fit unused space in the existing E*rows count allocation; no new buffers,
no expanded GPU heap payload, unchanged 18 KiB threadgroup operand slab.
Ordinary Full, short appends and decode retain M64.

Each output retains the same 32x32/K64 TensorOps descriptor and K64 update
order; the decoded shared Half slab feeds up to four live row products.
Clipping, F32 weighting, Half storage, down scatter and final reduction stay.
The shader retains the established mlx-serve MIT attribution; the already
inspected planner's 96-row rung motivates the initial hot-run cutoff.
The current code already skips inactive products, so this targets repeated
weight decoding across panels rather than claiming all padding is computed.

Predeclared gates: run the existing GPU differential in the test profile
with MTL debug and both policies, four IQ formats, guard/alias/hot/tail/empty
runs including 95/96/129; no new ignored test. Then same-`iter` interleaved
48K-root dispatch pairs, excluded first warmup pair, >=3 retained pairs,
complete bitwise endpoint/state/following-step, no overflow, and >=1.25x
combined routed GU+down GPU time. Report both components; credit logical
matrix work once across the disjoint M64/M128 families. Proceed to native
root/80K append and pp4096/8K controls only on success, then opposite-order
B3 before any promotion. Rejection removes the prototype and retains evidence.

### Adaptive expert guarded differential gate

M5 source `ee6e03913`, test profile, MTL debug, NI10/pressure0. The existing
single differential passes with both policies: **96 bitwise comparisons,
6,340,608 components**, all Half inner/F32 down boundaries, four formats,
mixed down formats, aliases, empty/hot/tail runs and 95/96/129 seams. All
allocation guards remain intact. This is a functional gate, not a speed claim.
`data/expert128-gpu-differential-summary.json` retains raw-log hashes.

Fresh `iter` SHA `4a520c2a2b0ede93a0d3e00d58b6774719bd8d186530134db6b6ff4ee5fb2c30`
is queued for real-root isolation with MID fixed on. The reduction also requires
zero M128 dispatches in baseline and positive M128 dispatches in every
candidate arm, and reports separate GU/down and wide-family GPU times.

### Header cache correction before expert isolation

The first real packet stops before the candidate dispatch: `v41_sorted_gu128_iq2_xxs`
is missing from `iter`. No performance or model-quality gate result exists.
The fresh test-profile binary contains the export and passes the differential;
the cached `iter` binary lacks it. Full binary hashes and export checks are in
`data/expert128-stale-shader-diagnosis.json`; failed raw packet stays separate
as `stage-expert128-ni10-*`.

Root cause: `build.rs` watched kernel directories but its per-source AIR key
hashed only the top-level `.metal` bytes, omitting included `.metalh`/`.h`.
Rerunning the script still accepted stale AIR. The key now includes the
transitive local include tree and emits explicit watches for those inputs.
Compiler/SDK/language/target/flags remain keyed; unresolved include spellings
fail rather than minting an incomplete key. No inference arithmetic change.

Three test-profile CPU regressions pass: nested header changes with preserved
mtime invalidate the key; cycles terminate and unresolved dependencies refuse
keys; all actual kernel tiers (including M5-only source) resolve and track the
sorted-expert header without needing that compiler. `check-contracts.sh` now
runs these regressions. Rebuild both profiles and repeat the complete paired
packet under a new tag, retaining the stopped attempt.

### Fresh include-aware artifact verification

M5 source `5ddfc5967`: the three cache CPU regressions and the guarded GPU
differential pass again at NI10/pressure0. The `iter` executable now contains
all eight M128 GU/down format exports; SHA
`6ac7052e35755912dde62e042a084c1ab661c29ce145f749c254cbf03949c259`.
The new complete paired packet is `stage-expert128-fresh-ni10-*`; do not combine
it with the stopped packet. Include watch emission is deduplicated per library
to keep build-cache metadata bounded without changing keys or dependency sets.

Full contracts pass at `cdc57c1f7`, including the three new cache CPU tests.
The initial run caught a four-command harness assertion after the new fifth
Cargo CPU target was added; update it to require that exact target while
retaining all offline/locked/unleased checks. No test was skipped.

### Adaptive expert isolated decision — rejected

Fresh include-aware source `5ddfc5967`, same `iter` SHA above, NI10/pressure0.
All eight complete endpoint/state/following-step signatures match bitwise;
no overflow. Baseline M128 dispatches are zero; candidate has 40 in each arm.
Three retained combined GU+down pairs: **479.459/747.938**, **480.931/746.326**,
**475.984/730.834 ms**. Mean **478.791 -> 741.699 ms, 0.64553x**, 0/3 positive.
GU **312.230 -> 488.373 ms (0.63933x)**; down **166.561 -> 253.326 ms
(0.65750x)**. Candidate wide-family GPU time alone is **545.987 ms**.

**Fail isolation; do not run native or serve.** Remove kernel/host routing,
workspace marker, product flag/scoped selector and prototype-only fixture
extensions; restore those files exactly to base. Keep the independent build
cache correctness fix, CPU regressions and all evidence. No threshold tuning
or unchanged rerun. `data/nice10-stage-expert128-summary.json`.

## Lever 8: wide cooperative PV and writeback, prototype

Default-off `QWEN_V41_ATTN_PV_WIDE` changes the actual H64/D512 PV tile from
D128/four SIMD groups/128 threads to D256/sixteen groups/512 threads. K32
updates remain in increasing-key order, with two serial dimension chunks
instead of four. Nominal cooperative destination storage falls from 64 to
32 floats per thread, unlike lever 7's extra independent accumulators.
The shared operands grow from 12 to **20 KiB**, checked against the device;
all allocated GPU payloads stay the same and no persistent buffer is added.

The candidate also performs F32-rounded normalization and inverse RoPE
writeback directly. This reuses the previously proven epilogue, with the
full mlx-serve MIT notice retained; the workgroup/descriptor shape is a
substantive change from the rejected D128 epilogue. Softmax and QK stay on
their existing paths. Eligibility is M5 Ultra, tensor64 H64/D512/R64 and
128..4096 logical rows. Other shapes and decode retain the existing path.

The existing tensor64 GPU differential is extended, without a new test or
ignore, to compare normalized wide PV bitwise against original tensor64 and
fused output bitwise against the original normalization/inverse-RoPE chain.
It covers K256/512/640, ragged visibility, query offsets and guards; it also
checks actual pipeline resources and production query-storage aliasing.
Local `iter` check and test-profile compilation/policy check pass.

Predeclared sequence: M5 functional differential first; then complete paired
48K-root and 1K@80K dispatch witnesses with >=1.25x combined
PV/normalization/inverse-RoPE GPU time, >=3 retained pairs after warmup,
all eight full endpoint/state/following-step signatures bitwise and no
overflow. Report PV separately as well. Native root/append and pp4096/8K
pairs follow only on success, then release B3 AB/BA before promotion.
No requantization, changed key order or approximate-score path.

### Expected Ghost process classification

The first wide-PV M5 build compiles successfully, then the functional helper
stops before running a test: the generic RSS>20GB guard calls the expected
`~/ghost/host-cache/.../headless/Headless.app/Contents/MacOS/shadps4` process
a model (PID53007, RSS28,692,432 KiB). No GPU differential, model load or
performance/quality result is claimed from this attempt.

The shared packet/serve check now distinguishes that exact Ghost executable
path pattern from model processes. Ghost stays in the complete process/load
record; real model tags and large unknown processes still refuse a load.
Three CPU regression cases cover the observed Ghost, model/unknown refusals,
lookalike paths and exact excluded-PID behavior; contracts include them.
The existing live-memory admission and thermal guards remain in force.
Preserve the stopped build/monitor and retry functional qualification.

### Wide PV functional gate — passed

M5 source `fceabe46b`, test profile, MTL debug and required-Metal mode:
all **15** K256/512/640 ragged/offset cases pass strict normalized and fused
inverse-RoPE comparisons, **13,434,880 components** total. Query-storage
aliasing and every guard remain intact. Actual 512-thread/20 KiB pipeline
resources are supported. NI10, pressure1 during settle and 0 during execution.
The expected large Ghost process is recorded. No model was loaded for this
test. `data/pvwide-gpu-differential-summary.json` retains hashes.

Fresh `iter` SHA
`4505aaad9a7f6521181b67e1dd18bee529d4a3016e8652d9a509fe93de2eeced`
contains both wide exports. `stage-pvwide-ni10-*` now queues the complete
48K-root and 1K@80K isolation packet; no performance/default claim yet.
Full contracts also pass from committed `fceabe46b`, including cache and
process regressions with all unleased/offline/locked checks intact.

### Wide PV isolation — both cells pass

Same verified `iter` SHA above, fixed MID=1, NI10 and pressure0 throughout.
`stage-pvwide-ni10` retains one declared warmup pair and three interleaved
pairs per cell. Every one of the eight endpoint/full-state/following-step
signatures is bitwise in each cell, with no dispatch overflow. Wide calls
are zero in every baseline and 160/root or 80/append in every candidate.

| Cell | Original PV + normalization + inverse RoPE | Wide fused output | Ratio | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 48K cold root, sampled late CED chunk | 221.307 ms | 132.631 ms | 1.66859x | 3/3 |
| 1,024 rows at 80K | 100.250 ms | 69.519 ms | 1.44205x | 3/3 |

Original root components are PV 118.857, normalization 22.552 and inverse
RoPE 79.899 ms; append components are 60.944, 11.431 and 27.875 ms.
Candidate dispatch time includes its epilogue, so this does not isolate a
matrix-only speedup. Both combined output stages exceed the predeclared
1.25x gate. Instrumented wall is not a native wall claim. Per-pair times,
raw hashes and active-dispatch counts are retained in
`data/nice10-stage-pvwide-{root,append}-summary.json`.

The native 48K-root/1K@80K packet is queued next, on the same `iter` binary
with the same one-plus-three pair protocol. PP4K/8K and fresh release B3
AB/BA remain required before promotion; the flag stays default-off.

### Wide PV native wall — both cells pass

`cell-pvwide-ni10`, same binary/source/MID setting as isolation, NI10:

| Cell | Baseline | Candidate | Mean ratio | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 48K cold root | 20,755.584 ms | 20,505.967 ms | 1.01217x | 3/3 |
| 1,024 rows at 80K | 1,014.148 ms | 995.341 ms | 1.01890x | 3/3 |

All eight full signatures per cell are bitwise. Root final retained pair
runs at allowed pressure2 and stays in the result; other root measurements
and all append measurements are pressure0. Each load first holds <=1 for
60 continuous seconds, including cooling from pressure2 before the append.
PIDs90322/97187 both exit0 with recorded niceness10. The complete load and
Ghost process record is in `cell-pvwide-ni10-monitor.jsonl`, SHA256
`929f6a43ce235920982eb778e9d0a13954c6117acff9d2b4ef13b1e450b20804`.

Root mean GPU command time is 16,986.630/16,667.505 ms; inclusive host
preparation is 1,153.273/1,213.818 ms. The stage gain becomes a modest
native gain, so release serving remains decisive. Per-pair walls and raw
hashes: `data/nice10-cell-pvwide-summary.json`. PP4096 and 8K comparison
controls are queued; fresh release B3 AB/BA follows. Default remains off.

### Wide PV PP controls and fresh release — passed

Same `iter`, MID=1, NI10/pressure0: pp4096 is **2,746.094 -> 2,689.707 ms,
1.02096x**; 8K CED is **3,709.183 -> 3,636.899 ms, 1.01988x**. Both retain
all three positive pairs and all eight bitwise signatures after the declared
warmup. PIDs3558/5227 exit0. Per-pair walls and raw hashes are in
`data/nice10-cell-pvwide-pp-summary.json`; monitor SHA256 is
`ff0c4cfba432e72ccaf751c7d31bf6c7e93d8a4f6f91698cfd496a9cbde4cf3a`.

Fresh M5 release source `659974115`, both wide export verification and build
pass, 3m17s. `qwen` SHA256
`a365a145f2d559102925386dc12b3e4df3539b21fcc8e601cc71c12f69b5d09f`,
tokenizer `eb94be08eb27fb5dfb0feaa423c83f958323970686e946b9f99e752fd7fc000e`.
`serve-pvwide-pair1w32` queues AB, same binary and cap32, three 48K agents,
fixed MID=1, only wide flag 0/1. Root/client TTFT and all equal-sized matched
short appends remain required; reverse BA follows a passing AB block.
No serving result or promotion yet. The prepared short-index extension is
absent from this release source.

### Wide PV release B3 AB — passes, reverse order still required

Same release SHA and cap32 above. Retain all three roots:
**36,946.636 -> 26,183.825 ms, 1.41105x** native prefill. Visible warm
TTFTs are 36.969/40.792/null versus 28.450/27.922/null, ratio **1.37944x**
over the two observed deltas per arm; no third delta is invented.

All **20/20** equal-sized matched short appends improve native prefill:
mean ratio **1.06164x**, median pair **1.06315x**. Client-visible TTFT mean
ratio **1.04912x**, median **1.04893x**, **19/20** positive. Delivered
aggregate text+reasoning rate is 76.183/79.650 tok/s; gap p99 .154806/.151904 s.

Both model processes record NI10. Baseline pressure2 is allowed and retained
(50.29 of 120 steady seconds); candidate maximum1. Baseline compiler overlap
is recorded, candidate compile-observed without steady overlap. Startup is
18.561/2.249 s, warmup 112.526/80.181 s, and ending load averages are
27.47/22.41/16.98 versus 13.55/18.49/17.43. These different backgrounds make
the reversed-order block essential; no sample exclusion or attribution of
the large root difference is made. Both before-load and after-warm settle
requirements pass. Pair completes within its 15-minute lease and child cleanup.

`data/serve-pvwide-pair1w32-comparison.json` retains all roots, matched calls
and allowed pressure. `serve-pvwide-pair2w32` queues BA on the same binary,
plan/cap and fixed MID. Default remains off pending that result.

### Wide PV release B3 BA — root gate fails; retire

Same release source/SHA/cap32 and NI10. Retain all three roots:
**20,949.577 ms baseline / 27,709.110 ms candidate, 0.75605x**. Visible warm
TTFTs are 21.269/20.898/null versus 29.768/27.536/null, ratio **0.73585x**.
The reversed served root gate fails; the candidate is not promoted.

All **15** matched short appends remain: native prefill mean **1.03405x**,
median **1.01845x**, 12/15 positive; client TTFT mean **1.02845x**, median
**1.02052x**, 11/15 positive. Delivered rate is 82.283/82.408 tok/s and
gap p99 .057620/.057088 s. The append positives do not override the required
root failure. `data/serve-pvwide-pair2w32-comparison.json` preserves all calls.

Both arms reach allowed pressure2 and stay in the result: 30.16 baseline
and 8.88 candidate steady seconds. Both record compiler overlap. Candidate
runs first, with startup 17.342 s and warmup 84.754 s; baseline startup
1.818 s and warmup 64.497 s. Ending loads are baseline21.63/20.35/18.30
and candidate17.78/15.78/16.28. These first-arm startup differences and
opposite root ordering are retained without a causal attribution or exclusion.
All settle rules pass; PIDs73900/77702 record NI10 and are cleaned up.

Remove the wide shader/epilogue, runtime selector/flag, perf override and
prototype-only GPU extensions. Restore the original attention tensor and
fixture files from the pre-prototype tree; keep raw evidence, reducers,
MID, cache correction and process safeguards. Short masked indexing is the
next M5 lever; cooperative experts follow. Before future root serving
qualification, predeclare symmetric process/cache warmup and record the
resulting disk-cache condition; do not resurrect this failed broad result.

## Lever 9: exact masked indexing for short deep appends — prepared, default off

The qualified MID lever starts at 512 rows, leaving the agent workload's
150..511-row appends on full-domain reindexing. Prototype
`QWEN_V41_MASKED_REINDEX_SHORT=1` adds ordinary **128..511** rows to the
same mask-skipping eligibility, retaining the existing depth threshold,
M5 Ultra and 32x128 index geometry, source-mask shape and global rollback.
MID=0 still restores the original 128K threshold. No shader arithmetic,
selection order, quantization, allocation or persistent memory changes.

The existing masked-MMA functional test covers allowed-score bits, partial
eight-key tails, per-query masks, all/none masks, offsets, guards and exact
selection. Predeclare its M5 test-profile execution, then 150- and 256-row
dispatch cells at 80K with >=1.25x index-stage time and three retained
interleaved pairs after warmup. Every full endpoint/state/following-step
signature must be bitwise; baseline masked calls must be zero and each
candidate must execute masked calls. Native short cells then require 3/3
wall gains. Include 128/511 boundary witnesses and 127/512/1024 plus the
48K-root inactive controls; report every measurement. Fresh release B3
AB/BA must improve matched short append prefill and client-visible TTFT
without a material root regression. Keep roots and all matched calls.

The served reducer additionally reports the potential 128..511-row subset
using each arm's actual native `prefill_rows` and depth. It preserves every
matched call and the original all-append/MID aggregates. Replaying four
archived complete AB/BA packets leaves every existing field exactly equal;
the new subset counts are 8/8/11/8. This is analysis validation, not a new
short-index performance measurement.

This preparation does not enter the pinned wide-PV binary or its serving
comparison. M5 short-append work starts only after the wide-PV decision and
uses the resulting qualified production path in both arms. Local format
and `iter` census checks pass; no short-append performance claim yet.

Full contracts pass from frozen `2669c0431`: 383 library CPU, 33 oracle,
54 benchmark, 175 CLI, 10 source, three cache and three process regressions.
The initial 30-second ENV generation cap expired before writing; contracts
correctly stopped on the stale reference. Generation with a 180-second cap
then completed, ENV was committed, and the fixed-tree rerun passed. No
test, policy or acceptance exclusion changed.

### Short masked index: functional and both isolation gates pass

M5 source `1930b69f5`, test profile, debug/required Metal: the existing
five-case masked-MMA test passes strict allowed-score bits, exact selection,
offsets/partial tails, all/none/per-query masks and guards. PID10794 records
NI10/pressure0 and exits0; no model is loaded. Hashes are retained in
`data/short-index-gpu-differential-summary.json`. All eight cooperative
expert exports compile but have not been functionally qualified; retired
wide-PV exports are absent from fresh `iter` SHA256
`d207776beaa50fc129dd1dbe64afe6a9531c7cb90604a11a53931b4423a72550`.

Same `iter`, fixed global mask/MID=1, cooperative=0, SHORT=0 outside the
scoped candidate override, NI10/pressure0. One warmup plus three retained
interleaved pairs per cell, all eight full signatures bitwise, no overflow:

| Cell at 80K | Full index scoring | Short masked scoring | Ratio | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 150 rows | 24.142 ms | 16.159 ms | 1.49402x | 3/3 |
| 256 rows | 40.823 ms | 27.117 ms | 1.50543x | 3/3 |

Every baseline has zero masked dispatches; every candidate has 24. The gain
comes from skipping rejected blocks, with unchanged allowed arithmetic;
it is not a claim of higher physical F32 throughput. Per-pair times and
raw hashes are in `data/nice10-stage-short-index-{150,256}-summary.json`.
Both isolation gates pass. Native paired cells are queued on the same
binary; boundaries/controls and release serving remain required.
Full contracts pass from frozen `1930b69f5`, including 383 library CPU,
33 oracle, 54 benchmark, 175 CLI, 10 source, three cache and three process
regressions. No acceptance exclusion changed.

### Short masked index: native short-cell wall passes

Same `iter` SHA/source and common flags, NI10/pressure0. After the declared
warmup pair, 150@80K is **391.239 -> 383.846 ms, 1.01926x** and 256@80K
is **501.584 -> 486.263 ms, 1.03151x**, both 3/3 positive. All eight full
signatures per cell are bitwise. PIDs79123/83339 exit0; monitor SHA256
`7a5488664b26b9b94817b205790ac583e5fbf440a9a8dc2a7e6899f52320c047`.
`data/nice10-cell-short-index-summary.json` keeps each pair and raw hash.
The 128/511 boundary packet is queued; 127/512/1024 and 48K-root inactive
controls follow, then fresh release serving. Default stays off.

## Prospective serving protocol: symmetric cache priming

The wide-PV blocks record large first-arm startup/warmup differences in
opposite orders. Keep their failed broad result. For subsequent levers,
`v41_prefill_serve_primed.py` runs **one arm per <=15-minute lease**. Each
arm first completes the same three 48K roots in an unmeasured process via
`serve_realistic.py --warm-only`, waits for that model to exit, then loads a
fresh measured process. Measured roots therefore have empty process/KV
state and primed file/pipeline caches. Label that cache condition explicitly;
do not combine its roots with the earlier unprimed blocks.

Each load keeps the continuous <=1/60s settle, permits 2 and aborts at 3.
The measured process also settles after warming, then uses the existing
120s steady plan, cap32 and three agents. The measured window body is
textually unchanged after extraction. Both priming and measured processes
record exact PIDs/niceness and must be gone before another load. Only
measured roots/calls enter comparisons; priming artifacts remain separate.

The controller binds binary/tokenizer/corpus/model path, plan, harness and
controller hashes, common feature environment, order and cap across both
arms. It refuses duplicate/out-of-order arms or overwriting evidence. Run
AB and BA prospectively. The 24 CPU harness checks pass, including new
priming cleanup/abort and pair-identity checks; no real model runs in those
tests. M5 execution of this protocol remains pending the next native gate.

Full contracts pass frozen `5f69c6098`, including the cache/process
regressions and unchanged acceptance inventory. The separate Python harness
suite passes all 24 cases. No measured M5 run uses the priming controller yet.

## Lever 10: cooperative M32/M64 experts — prepared, default off

`QWEN_V41_EXPERT_COOPERATIVE` spreads the existing sorted native-IQ panels
over **eight cooperative SIMD groups / 256 threads**. GU uses M32/M64 x N64,
down M32/M64 x N128, retaining K64 updates and the original dequantizer.
Runs with <=32 live rows use M32, preserving the short-run work reduction
needed by 1K appends (mean 16 routed rows/expert). Plans remain 64-row panels.
This changes the cooperative descriptor/workgroup and writeback; it does
not restore the rejected M128 independent-accumulator kernel.

Direct accumulator writeback replaces the staged output passes. Volatile
F32 activation and weighted values retain the two storage rounding
boundaries before half; down remains F32. Nominal full-panel destination
storage is 32 floats/thread versus the original 64, without a physical
register-allocation claim. Shared operands remain **18 KiB**; no tensor,
plan or persistent GPU allocation is added. Eligibility stays on the
existing M5 Ultra sorted 1024..4096-row/config/format tier and its rollback.
The existing header's complete mlx-serve MIT notice remains in place.

Local format, `iter` census and test-profile type checks pass. The existing
GPU differential now runs both modes across four formats, two down types
and hot/empty/short/tail histograms: 32 cases, 4,227,072 strict components,
plus plan/packing/alias/guard checks. These are required M5 comparisons,
not a claimed result; no new ignore or test name is introduced.

Predeclare M5 test-profile functional qualification first. Then root/1K@80K
dispatch witnesses must each show **>=1.25x combined GU+down**, three
retained interleaved pairs after warmup, all eight full signatures bitwise,
no overflow, baseline cooperative calls zero and every candidate active.
Report GU/down separately. Native root/append, PP controls and fresh
release B3 AB/BA follow on success. Failure retires the candidate. It stays
off and out of the pinned wide-PV release. Model-free M5 functional
qualification may run while the independent short-index gates progress;
cooperative performance comparisons follow the short-index decision and
use the resulting qualified path in both arms. No performance or default
claim yet.

The first full contracts run stops at the identity-policy check: the two
new explicit M5 Ultra restrictions need scoped registry entries. Record
only those exact path/token pairs with removal triggers tied to additional
device-class functional/stage/native/serve qualification. The scanner,
acceptance inventory and numerical bounds stay intact. The redundant device
name in an error message becomes a generic unqualified-device diagnostic.
A new fixed-tree contracts run follows; no GPU result is inferred.

That rerun catches formatting in the diagnostic edit before execution.
Apply rustfmt, then rerun all contracts after the independent wide-PV
retirement is committed. Neither stopped run changes acceptance exclusions.

## Wider-chunk continuation boundary

The existing [8K CPU ledger](../2026-10-29-v41-encoder8k-cpu-result.md)
already prices one 8K encoder against two co-live 4K chunks: its synthetic
Metal quote falls about 191 MiB and CPU rises 24 MiB. Those figures use
assumed heaps and 8K/16K capacities, not this artifact's live three-session
admission. Runtime remains clamped to 4K and decoder replay to 2K. Opening
execution needs explicit admitted 8K geometry, actual route populations,
projection/expert lineage proof and the same numerical/wall/serve gates.
No 8K execution or memory-fit claim is made by this lane. The census and MLX
comparison identify that continuation without bypassing its existing guards.

## Validation and pending work

`scripts/check-contracts.sh` passes after MID promotion and the feature-scope
fixture at `1a206f6b2`:
383 library CPU, 33 oracle,
54 benchmark, 175 CLI and 10 source tests, with no failures or ignored execution.
The earlier attempt failed source provenance because editing continued during
the check; it was rerun from a fixed committed tree. No test was disabled.
The census example type-checks in `iter`, including `perf-ablations`;
rejected price examples are removed.
Regenerate ENV and rerun contracts after final lever decisions. No roadmap edit or
merge has occurred. MID is promoted only after the recorded gates; the other
active candidates remain off.

After PV removal, full `scripts/check-contracts.sh` passes at `8c2d0e7ac`:
383 library CPU, 33 oracle, 54 benchmark, 175 CLI and 10 source tests;
no failures or ignored execution. The retained publication pending-byte
correction is included. The removed PV selector is also removed from probe
argument validation; final source checks follow the final publication decision.

After publication retirement, `scripts/check-contracts.sh` passes at
`df8a5f05a`, including 383 CPU/33 oracle/54 benchmark/175 CLI/10 source tests.
The adaptive expert prototype is subsequent work and needs its own checks.

Wide-PV prototype contracts pass from committed `b6494c143`, including
383 library CPU, 33 oracle, 54 benchmark, 175 CLI, 10 source and three cache
regressions. No ignore or acceptance exclusion changed. M5 functional
qualification remains queued; no model performance result yet.

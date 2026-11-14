# V4.1 compact + exact M2: composed serial gate fails

Bounded stage started 2026-11-14 02:26 UTC, hard stop 04:26 UTC. The lead
authorized restoring compact for composition; its standalone all-cell failure
in [the previous stage](2026-11-13-v41-attention-serial.md) stands unchanged.
Audit findings remain hypotheses until actual arguments, ranges and dependencies
are verified. This experiment changes only index/selection work in the serial
span; it does not change attention preparation or finish/post work.

## Candidate and frozen rule

Compact turns the 2,048 kept eight-key blocks into chronological block IDs,
then scores those blocks. M2 shares a loaded key matrix across two independent
F32 query accumulators. Each query keeps the incumbent 16 K updates, ReLU,
weight multiplication and four-head-group sum. Adjacent queries can require
different shifted windows for a partial eight-key tail; those windows load
separately. Odd rows, nonzero query/output offsets and the top-k boundary are
covered by the strict differential. Neither component changes input precision
or adds persistent tensor memory. The compact IDs use existing pair scratch.

Independent experimental controls:

- `QWEN_V41_MASKED_INDEX_COMPACT=0` restores full-domain masked launches.
- `QWEN_V41_INDEX_MMA_M2=0` restores the one-query dense-index MMA.

Both remain default-off during qualification. The paired probe scopes the two
flags together with `IndexCompose(false/true)` on a single archived iter
executable. Its prefix uses production with both experimental flags zero.
These scoped probe arms do not alter the production rollback semantics.

Before the first composed arm, `plan.json` freezes the same rule: at each of
256/1K rows x 80K/128K depth, AB warmup then three BA/AB/BA measured pairs.
The geometric mean of paired baseline/composed summed
`prefill_attention_serial` GPU intervals must be **>=1.25 in every cell**.
Every logits/causal-state/following-decode signature must be bitwise. No pairs
are omitted. A failure removes both components; a pass proceeds to native
256/1K @80K on the same binary with warmup and three pairs, then stops.
No serve qualification or default-on promotion is part of this stage.

## Implementation and pre-gate compiler work

The kernels derive from this repository's incumbent index MMA and compact
implementation under its MIT/Apache-2.0 licenses, with provenance in the shader.
No external source was adapted. Source `4db569169` restores compact and adds
M2. Its strict GPU witness passes 14 cases and **413,968,005 score-bit
comparisons**, including canaries, real 256/1K populations at 80K/128K for
ratios 1/2, odd rows, nonzero offsets, partial tails and top-k crossings.
The restored compact witness also passes.

The first M2's diagnostic GPU timings under Metal validation were only about
0.032x the one-query kernel across all eight large shapes. These are construction
diagnostics, not a composed-cell gate. Its artifacts and pins remain under
`attempts/4db569169/`. Compiler IR for both one-query and M2 contains private
matrix arrays; the IR alone does not establish the final register/spill cause.

Source `32cbcb219` explicitly unrolls matrix accesses and moves tail-window
branches out of the shared inner K loop. It is rebuilt and pinned separately
before another strict witness. The gate rule is unchanged; no composed model
arm used the initial slow implementation.

The unrolled register version remains about 3% slower in diagnostic timing.
Construction source `c0aeb597f` instead stages a padded F32 key panel for 16
simdgroups (512 threads), keeping only one query matrix set per simdgroup.
It is also bitwise but slower. Source `8a1bc007c` returns to register M2 and
caps compiler resources at the actual 256-thread launch; that does not change
throughput materially. Every version passes the same 14-case strict witness
and the restored compact differential.

Before any composed model arm, the implementation-selection rule chooses one
fixed M2 by the highest geomean of its eight diagnostic large-shape ratios.
It selects **`32cbcb219`**. This is a numeric selection among diagnostic points,
not evidence that its tiny difference from the resource-capped version is
significant. There is no per-cell implementation choice. The active Rust and
shader files at tip `2f91e67d9` match this selected source byte-for-byte.

| construction source | M2 layout | diagnostic geomean vs incumbent |
|---|---|---:|
| `4db569169` | register M2, inner branches | 0.032296x |
| `32cbcb219` | register M2, explicit unroll | 0.973072x |
| `c0aeb597f` | padded shared keys, 16 simdgroups | 0.649629x |
| `8a1bc007c` | register M2, max 256 threads | 0.972990x |

[implementation.json](2026-11-13-v41-index-compose/implementation.json) keeps all
32 diagnostic points. They use the existing helper's median of five command
buffers after warmup, three encodings per buffer, with Metal validation enabled;
they are not a standalone performance gate or the composed gate. No composed
model sample used an unselected implementation.

Selected iter SHA256 `fcd30d2c126a5dc22463467c8fa99591e00e6838caed4961895f7ba300d8b719`;
test `e44723b1351e41be161a71d013634da3db0b035f27819eac6d0bc03a1a5f01be`.
Both profiles have main metallib `49e4c119a20cfa2b8181e699a51cf954ae1faeb792fc1a9137ae1228771922e9`,
tensor `43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c`,
research `be63c8ae9332529da45c21328333191834bd767917f5f9bc55d94eb3702b557f`.
Both arms use this exact archived iter executable under
`attempts/32cbcb219/pins/iter`, including after later construction builds.


## Measurements

Raw artifacts: `metagross:~/qwen-runs/2026-11-12-v41-prefill/attention-compose/`.
The worktree is `~/code/qwen-llm-v41-index-compose`, separate from main.
Builds use nice 10 under waiting leases of at most 15 minutes; measurements use
nice 0, verified by `ps -o nice=` and `getpriority`. Python forwards the lease FD
with `pass_fds`, checks for other resident models and records background load.
Each load settles <=1 for 60 seconds, permits 2 and aborts at 3. Executables
are copied out of `target/` and hashed before use; pins are kept until the gate
closes. Per-arm process disk-read bytes remain in every cell record.

**FAIL in every cell.** All 32 logits/causal-state/following-decode endpoints
are bitwise. The rule and every measured pair stand.

| depth / rows | baseline serial ms | composed serial ms | three paired ratios | geomean |
|---|---:|---:|---|---:|
| 81,920 / 256 | 40.073333 | 35.157846 | 1.141725 / 1.137101 / 1.140616 | 1.139813 |
| 81,920 / 1,024 | 122.662736 | 102.021250 | 1.202014 / 1.202292 / 1.202670 | 1.202325 |
| 131,072 / 256 | 61.297360 | 52.976320 | 1.159158 / 1.154978 / 1.157084 | 1.157072 |
| 131,072 / 1,024 | 194.720236 | 158.257292 | 1.233113 / 1.233571 / 1.224545 | 1.230403 |

Every measured append window has **0 process disk-read bytes**; warmups
have disk reads and are retained in the raw/summary arms. Thermal pressure
is 0 in every observed sample; all children are nice 0. One-minute host load
during the model packet ranges 10.70–29.99, with Ghost background work recorded.

The sum is over the eight active global layers, 2/8/14/20/24/28/32/36.
Every active interval is present and valid in every arm. The 32 structurally
empty serial stages have no GPU timestamps; those placeholders are not
missing measurements of active work. No active interval or pair is dropped.

Full arms, layer times, disk bytes and signatures:
[gate.json](2026-11-13-v41-index-compose/gate.json).

Informational stage-profiled append wall means (baseline/composed):
451.00/450.46 ms at 80K/256, 962.89/949.63 at 80K/1K, 466.21/461.29
at 128K/256, and 1,035.14/993.51 at 128K/1K. These are not a native wall
qualification or a basis to override the serial-span failure.

Native cells and serve were **not run**, following the failed-gate branch.
Both compact and M2 implementations, their scoped experiment variants, added
GPU tests and rollback knobs were removed. Active Rust/shader files return
exactly to pre-stage `1e7bb4baa`; the prior audit/probe fixes remain.
The selected code is archived at `backup/v41-index-compose-measured`.

The original compact-only failure is unchanged. If revisiting this span,
the next exact experiment should target radix's repeated masked `-Inf`
histogram updates: aggregate identical-bin integer counts before atomic
updates, retaining the old threshold, NaN and chronological tie behavior.
That is a recommendation, not an implemented or measured improvement here.


## Validation and closeout

- Local iter compilation and M5 iter/test builds pass. Every construction's
  strict differential passes 14 cases with 413,968,005 score-bit comparisons;
  selected-source real-model signatures are 32/32 bitwise. No ignored test
  was added. No approximate arithmetic or requantizing was introduced.
- Final committed-tree contracts output is retained as `contracts-final.log`
  in the run directory. ENV is regenerated after withdrawal (830 knobs).
- At gate closure, all **eight** archived executable hashes across the four
  constructions were reverified. Both selected and unselected pins, the M5
  target, all four transfer bundles, full-module diagnostic dumps and the
  M5 worktree were removed. Small selected-function IR and raw evidence remain.
- `cleanup.json` confirms **12/12 observed model/test exits**, no remaining
  task processes and no task-process kills. PID reuse is recorded separately
  without signaling unrelated processes. Closing local/M5 process checks
  accompany the contracts log. No pushes or merges; no roadmap edits.
- SRV-2K0Y start note: `DKJ9F4NVH2DY8C4A`; the closing note records the final
  committed tip and verification result.


# 2026-11-11 — GLM layer-sliced append: exact, without another weight pass

Status: design and native probe; no production scheduler change. Branch
`research/layer-sliced-prefill`. GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra, 49152
committed tokens, 65536 capacity, normal Fast/Half 2048-row workspace, real
`code.txt` corpus from the three-agent harness. Build: `nice -n 10 cargo build
--profile iter -p qwen-llm --features perf-ablations --example layer_slice_probe`.

Probe: `crates/qwen-llm/examples/layer_slice_probe.rs`. Raw:
`metagross:~/qwen-runs/2026-11-11-layer-slice/{sweep-w3.jsonl,policy-w2.jsonl}`;
matching stderr, `build.log` and source/data hashes in `metadata.json` are
alongside. Eight paired repetitions, order
alternates, one canonical lease per invocation, requested holds 12 minutes.
The first sweep held 4m41s, waited 0s, and recorded thermal pressure **0**
throughout. The policy sweep held 5m25s, queued for 26.5s, and recorded thermal pressure
**0–1**. Spot host load samples were approximately 8–20; GPU clocks were not
sampled. The M5 worktree was removed after both runs.
An earlier launch stopped before model load because the fixture path was unset.

## Existing yield granularity

GLM's `QWEN_SERVE_GLM_PREFILL_FAIR` defaults on. In
`serve/batching/driver.rs`, fair load preserves `runtime.rows(run)`; the
rollback selects 512 rows. Neither means a layer or a time quantum. The GLM
adapter attaches the request's session, calls `prefill_run`, then detaches it.
That call invokes `prefill_packed_with_checkpoint` over a row segment; the
packed forward encodes **all 45 layers and the head before returning**.
Cancellation/observation callbacks run at whole row-chunk boundaries.

`QWEN_SERVE_GLM_DECODE_PERCENT` defaults to 10. The driver earns decode credit
only **after** `runtime.prefill` returns:
`min(prefill_elapsed * percent / (100 - percent), 1 second)`. It then spends
that credit on whole decode rounds/bursts. Raising the percentage changes
service after a chunk, not the longest gap inside it. The 512-row rollback
also changes the row partition and pays another per-forward expert weight
stream; it is not an exact layer split.

For a one-chunk warm append, prefill is already finished when credit is
earned, so it cannot provide service inside that append.

V4.1 shares the elapsed-credit policy. Live CED can yield between encoder
row submissions, normally 4096 rows, but its final decoder reconstruction is
one synchronous block. Its native dependency halo is 2541 rows; 53/54 real
warm appends are shorter and cannot enter CED regardless of the serve gate.
The current 8192 gate therefore is not the cause of those stalls. The
sub-4096 CED experiment changes the row partition, fails bitwise equivalence,
and still leaves a >1s final slice. See
`2026-11-11-v41-live-ced-threshold.md` on `perf/v41-live-ced-threshold`.

Consequently, the observed 1–3s gaps are consistent with the scheduler:
normal decode service resumes only after a full append or chunk. The
separate post-idle residency penalty can further lengthen the first append;
this probe prices warm mixed work and does not claim to bound cold faults.

## Suspension and isolation

A boundary **after the complete FFN post/mHC operation of a layer** is enough.
The next layer consumes only `packed.residual[0]`. The loop also needs the
next block index, MLA ordinal, fixed base position, token count and row plan.
No attention/FFN intermediate from the completed layer is live.

- MLA scatters the append's F16 latent rows into that layer's cache at the
  unchanged base position. Indexer append updates that layer's pending ring
  and pooled keys. Sparse visibility and selector status are prepared once
  for the original rows. Later layers have independent cache regions.
- KDA completes the whole original row recurrence and conv-tail update for
  that layer. No recurrence is cut into new row groups. Its state is private
  to that sequence and layer.
- Completed layers have advanced state while later layers have not. That
  session must remain exclusively in a prefill phase: no decode, snapshot,
  cache publication, workspace release or new append on it. Publish its new
  position only after the final layer/head and ordinary status validation.
  If a request is cancelled mid-layer-forward, discard the partial session
  or finish it before reuse; the old position is not a rollback of KDA state.
- `Glm5NextBatch::step` takes an explicit list of other sessions. It borrows
  the **first decoder's** packed workspace, and each decoder's own MLA/KDA/
  indexer/selector regions. Batch logits belong to the batch. It has no
  reference to the parked appender. Weights are shared; writable sequence
  state and activation buffers are not. The probe checks both directions of
  isolation against the equal-work unsplit arm.

The live residual is F32 with four mHC streams: **rows × 4096 × 4 × 4 bytes**,
not rows × hidden × all 45 layers. One parked append retains 2.00 / 21.375 /
94.3125 MiB for 32 / 342 / 1509 rows. The existing two 2048-row residual
buffers reserve 256 MiB in total; those and all other prompt scratch already
exist. The probe allocates **zero additional activation bytes**. Multiple
parked appends would multiply retained workspaces; the proposed first policy
has just one appender and continues the other decoders.

### Commands and the causal executor

Packed prefill currently starts a serial `KernelEncoder::begin` and submits
one command per row chunk. The probe ends the encoder, commits and waits after
selected complete layers, runs a separate independent decode batch on the same
queue, then creates a fresh prefill command. Row geometry, kernels, operands,
MLA ordinal, reductions and head execution are unchanged. Every append layer
still runs exactly once. No command for the next prefill slice is queued ahead
of the intervening decode.

Independent batched decode uses `begin_causal` when dependency waves are on;
its recorded binding graph is local to the encoder and its scopes finish at
layer boundaries. Metal barriers order waves within a command. Those scopes
are **not host/scheduler yields**: a real yield must finish the scope and
submit/drain the command. This experiment leaves decode's normal causal path
on and does not split or replay a half-finished scope.

The diagnostic hook is compiled only with `perf-ablations`. It retains the
normal poisoned flag until the entire append validates. Its callback cannot
borrow the appender and may only advance other sessions. It is a synchronous
pricing hook, not the production resumable cursor API.

## Requested sweep: three decoding members plus the appender

Three *other* members means four live sequences. This is the requested stress
case; the following section checks two other members for three agents total.
Layer endpoints are `floor(i * 45 / s)` for `i=1..s`, including the head in
the final slice. `s=1` is one command and no intervening round.

Every pair restores the same 48K root, completes an initial decoder round,
then starts the append. The split arm runs `s-1` decoder rounds **between**
slices. The baseline runs the original single-command append, then the same
`s-1` rounds. One common final round closes the last append gap in both arms;
that round is excluded from TTFT and equal-work GPU/wall totals. Thus the
last slice cannot disappear from the gap statistic. Decoder inputs are the
same teacher-forced corpus tokens in both orders.

GPU work is the sum of Metal `GPUStartTime`/`GPUEndTime` spans, including all
append commands and those same `s-1` decoder commands. No per-stage profiler
or timestamps that change kernel selection are enabled. TTFT is time through
append logits readback, before sampling/SSE; gaps are completion-to-completion
per decoder. All members of a round share its completion timestamp. These
are native gap samples, not a served HTTP A/B or a statistically independent
sample per member.

TTFT columns are paired-arm medians in ms; gap columns are empirical pooled
p99 / maximum over eight repeats; GPU delta is the mean paired ratio against
the unsplit append **plus the same rounds**.

| append rows | slices | unsplit TTFT | sliced TTFT | other-member gap p99 / max | total GPU delta |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 300.3 | 300.3 | 328.6 / 328.6 | +0.01% |
| 32 | 2 | 299.9 | 326.9 | 184.1 / 184.1 | -0.02% |
| 32 | 4 | 300.1 | 380.7 | 107.2 / 107.2 | +0.15% |
| 32 | 8 | 300.2 | 486.7 | 74.4 / 74.4 | +0.05% |
| 342 | 1 | 575.2 | 575.0 | 602.0 / 602.0 | -0.04% |
| 342 | 2 | 575.1 | 602.1 | 330.3 / 330.3 | +0.03% |
| 342 | 4 | 575.0 | 655.7 | 185.2 / 185.2 | +0.06% |
| 342 | 8 | 575.2 | 762.5 | 114.2 / 114.2 | +0.07% |
| 1509 | 1 | 1559.7 | 1559.8 | 1603.1 / 1603.1 | +0.01% |
| 1509 | 2 | 1559.9 | 1586.8 | 851.4 / 851.4 | -0.10% |
| 1509 | 4 | 1560.3 | 1641.6 | 456.7 / 456.7 | -0.06% |
| 1509 | 8 | 1560.0 | 1748.6 | 266.9 / 266.9 | -0.02% |

**Every exactness gate passes in all 96 pairs:** all append logits bitwise
against a separately executed ordinary `prefill_packed`; all captured cache
bytes identical (MLA latent/pending/pooled, KDA conv/state); one common
post-append decode bitwise; every decoder round's logits and final decoder
cache bytes bitwise against its unsplit equal-work pair.

There are only 8/16/32/64 gap samples per table cell, so nearest-rank p99 is
also the maximum. This is a repeatable append-local bound in this sample,
not a prediction of the 30-second harness's percentile mixture. The 1509-row
warm append is ~1.56s here versus ~1.68s in the earlier chunk-price run;
paired comparisons, not cross-run absolute differences, establish the cost.

The GPU price is within -0.10% to +0.15% across all cells. TTFT grows by the
inserted rounds (about 27ms each). The 200ms per-forward floor is **not**
repaid at a layer boundary. Eight slices alone leave the large append at
267ms; a smaller time quantum needs more layer submissions, not smaller row
chunks.

## Three-agent policy calibration

The two-decoder sweep additionally prices 16 and 32 slices. All **96 pairs**
pass the same four exactness gates, including under thermal pressure 1. This
brings the total to 192 passing pairs (384 timed arms). The selected settings
and large-append alternatives are:

| append rows | slices | unsplit TTFT ms | sliced TTFT ms | gap p99 / max ms | total GPU delta |
|---:|---:|---:|---:|---:|---:|
| 32 | **4** | 299.4 | **367.4** | **102.8 / 102.8** | +0.01% |
| 342 | **8** | 578.7 | **750.8** | **112.7 / 112.7** | +0.13% |
| 1509 | 8 | 1567.9 | 1742.7 | 265.5 / 265.5 | +0.00% |
| 1509 | **16** | 1565.0 | **1931.3** | **151.5 / 151.7** | +0.01% |
| 1509 | 32 | 1565.2 | 2311.5 | 121.8 / 122.2 | -0.03% |

For the selected settings, median equal-work GPU totals (baseline → sliced)
are **359.84 → 359.75ms**, **723.88 → 725.15ms**, and
**1874.91 → 1873.59ms**. Median paired equal-work *wall* differences are
+1.0 / +4.4 / -10.0ms. The GPU ratios in the table average paired ratios,
so rounding the separate medians need not reproduce them exactly.

All twelve two-decoder settings, including overslicing short appends, have
mean paired GPU deltas between -0.10% and +0.47%. Overslicing has a clear
TTFT price anyway: 32 rows in 32 slices take **1011ms** instead of 299ms,
although their gap max is 38.7ms. Moving 1509 rows from 16 to 32 slices saves
only 30ms of maximum gap and adds about 380ms TTFT. Start with 16 there.

## Recommendation and production design

**Build the GLM resumable prefill cursor.** The native gates pass: full bitwise
state/output equivalence, no new activation allocation, and essentially
unchanged total GPU work. Keep this branch as the design/pricing artifact;
there is no serve policy to enable in it.

Use a **nominal 100ms of prefill GPU work per slice**, followed by **one
batched round of all eligible other decoders**, only while another decoder
is ready. Keep the original row plan and all kernel dimensions. The measured
starting points are 4 / 8 / 16 slices for 32 / 342 / 1509 rows, or roughly
11 / 5–6 / 2–3 layers per slice. Unequal layer costs make the largest measured
slice about 127ms, so the resulting native gap is about 103 / 113 / 152ms.
Use row/depth-specific layer cost estimates and recent completed-slice times
to choose boundaries; a nominal target is not a hard 100ms bound on an
unseen append. Keep the head in the last slice's budget. With no decoder
ready, finish the normal unsplit command.

The production change needs these explicit semantics:

1. A prefill cursor owns the fixed chunk's base position, tokens/row count,
   next layer and MLA ordinal, plus exclusive access to its existing scratch.
   Stage token IDs and sparse controls once at begin, not on every resume.
2. A slice ends after FFN post, commits and completes its command, then returns
   to the ordinary scheduler. No ahead-of-decode prefill command is submitted.
3. The driver excludes the appender from the batch, runs one other-member
   round, and resumes the cursor. Integrate this service with existing credit;
   do not pay the old post-chunk decode credit a second time. The current
   strict 10% budget is insufficient for a round every ~100ms: the measured
   two-member round costs ~23–25ms. Make this minimum service explicit.
4. Only final completion validates every route/selector status, publishes the
   new sequence position, exposes logits, updates history/canonical metadata,
   and permits checkpoint capture or decode admission. A partial-layer cursor
   is not a committed row prefix. On cancellation/failure, discard the partial
   session unless it is deliberately driven to completion under the same
   fairness policy. Do not pretend that restoring the old position restores
   recurrent state.

Expected responsiveness for these **warm** three-agent appends is native
append-local p99/max **~105–155ms**, plus sampling/transport jitter, versus
~0.33 / 0.60 / 1.60s unsplit. The appender pays **+68 / +172 / +366ms TTFT**
(about 23–30%), since other agents receive tokens during its append. This is
a deliberate latency tradeoff, not extra append GPU compute. Cold-idle
residency faults and arbitrary longer appends were not priced and are outside
that maximum; the previous cold-idle finding still needs separate treatment.

Expected aggregate delivered throughput is **roughly 75–79 tok/s** against
the standing 76.3–79.4, approximately flat to 2% lower, **an inference pending
a served A/B**. Equal-work GPU timing alone cannot prove delivered tok/s:
with three total agents, inserted rounds have two members, whereas rounds
after append completion can include all three. A conservative fixed-output
budget adds `k * (D2 - 2/R3)` per append, plus measured split overhead, where
`D2 ≈ 23–25ms`, `R3 ≈ 107–108 tok/s`, and `k = s-1`. At a 25–29% prefill
share this is about a 1–2% wall penalty for the selected slices. The native
rounds also replace some existing service on multi-chunk appends. Do not
promise a throughput gain from reordering identical GPU work.

The next implementation gate is the same 30-second three-agent serve harness:
compare delivered tok/s and actual streamed gaps, appender TTFT and partial
prefill accounting, with the native bitwise gate retained. The present probe
supports building that cursor; it does not substitute for the served gate.

## Validation

- Local and M5 `nice -n 10 cargo build --profile iter` for the feature-gated
  example passed. Source SHA-256 hashes match between the two machines.
- All 192 paired GPU probes passed all four bitwise gates. No ignored tests
  were added.
- `nice -n 10 scripts/check-contracts.sh` passed, including feature-off builds,
  CPU acceptance inventory and the comparison tripwire, in the test profile.
- Both M5 leases stayed below six minutes; raw output and metadata remain in
  the run directory. The M5 worktree was removed; nothing was merged.

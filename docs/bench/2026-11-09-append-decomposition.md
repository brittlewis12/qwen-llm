# Agent append roofline decomposition, 2026-11-09

Production base **2bad80c4416b22ce4821d99c6239cb05f82f173e**, M5 Ultra,
unchanged **Flash-Next GSQ-RCO IQ3_S** / **GLM-5.3 UD-IQ3_XXS**.
Scratch probe/installation recipe **3023708c**, three retained native samples
per cell after one excluded warm call. Real Rust source, one 65536-token
prefix/model, restored native snapshot before each append. Timing includes
full endpoint logits/status handling; prefix, restore and hashes are outside.
No runtime or shader optimization is committed.

## Ordinary append and proper roof

Use **max(F / 131.676 TFLOP/s, B / 1175.47 GB/s)**, decimal units. These
are the same M5's measured warmed F16/F32-accumulate tensor and loads-only
bandwidth calibrations from November 6, not advertised peaks. `F` counts
useful learned matmuls and selected QK/PV/index products. `B` counts each
actual touched native expert once, other executed native weights once, and
minimum architectural state/cache/index payload. Extra scratch/repeated
traffic and tile padding are excluded from this optimistic lower bound.
Tiny embedding/PLE row gathers are omitted, never charged as whole tables.

| Family | Append | Native ms | Weight GB | Mean touched experts/layer | Compute / byte floor ms | Global roof ms | Useful roof fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fn | 400 | 580.2 | 37.05 | 334.0/512 | 39.1 / 31.8 | 39.1 | 6.7% |
| fn | 2000 | 1522.4 | 48.91 | 454.3/512 | 195.6 / 41.9 | 195.6 | 12.8% |
| glm | 400 | 1089.3 | 106.41 | 260.0/288 | 111.3 / 90.8 | 111.3 | 10.2% |
| glm | 2000 | 2341.8 | 115.04 | 282.7/288 | 556.9 / 98.2 | 556.9 | 23.8% |

At 400 rows, routed-expert weight streams alone take **27.8 ms FN / 84.4 ms
GLM** at the bandwidth roof, versus **14.3 / 51.4 ms** useful expert compute.
Thus the expert phase is bandwidth-bound even though the global aggregate
roof is compute-bound. Summing per-projection `max(F_i/peak,B_i/BW)` gives a
stronger **incumbent-stage** learned-matrix floor; that is used in the additive
ledger below. It assumes the existing phase order, not arbitrary future overlap.
The full GGUF size is also the wrong byte count: FN's large gathered PLE table
is not streamed in full by an append.

Fresh 1-GiB checked load controls reach **1166 GB/s FN / 1165 GB/s GLM**.
FN's short sequence ramps from 525 to 1166; its prespecified retained median
is **1035**, versus **1162 GLM**. Keep every sample. Because that FN median
contains startup ramp, the roof retains the established warmed **1175.47**
calibration for both families. Using either observed retained median leaves
all four global roofs compute-bound. Tensor peak is reused, not remeasured.

## Additive diagnostic allocation

Milliseconds; rows sum to the native mean. This is a **model**, not exclusive
hardware attribution. Learned floor excludes attention arithmetic; context
attention is charged separately, avoiding double-counting the whole-call roof.

| Family / rows | Learned compute/byte floor | Context attention | Non-GEMM | Launch/serialization model | Host nonoverlap | Kernel excess | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fn / 400 | 50.6 | 82.0 | 46.8 | 8.9 | 4.2 | 387.7 | 580.2 |
| fn / 2000 | 183.7 | 347.7 | 168.4 | 37.8 | 8.4 | 776.4 | 1522.4 |
| glm / 400 | 131.3 | 162.2 | 62.7 | 5.1 | 2.3 | 725.7 | 1089.3 |
| glm / 2000 | 489.7 | 789.6 | 264.8 | 60.7 | 5.7 | 731.5 | 2341.8 |

Per-dispatch profiling splits encoders. Counter starts can precede dependency
completion, so summing isolated spans would double-count waits. Partition the
serialized diagnostic into uncovered transition gaps and nonoverlapping
completion increments. Attribute profile-minus-native GPU inflation to those
gaps first; distribute the remaining native kernel budget by interval-core
shares. Launch/serialization is the **maximum**, not sum, of the residual-gap
allocation and a checked dependent-empty-dispatch curve. Host is native wall
minus native GPU. Kernel excess is the residual over the learned stage roof:
quant unpack, padding, extra traffic, occupancy and dependency stalls remain
mixed there. This cannot establish an exclusive barrier cost or recoverable
milliseconds. The script retains the model inputs and raw kernel aggregates.

The critical perturbation is **GLM 2000**: native GPU **2336 ms**, profiled
**2822 ms**, with **547 ms** uncovered gaps. About **486 ms** is measurement
inflation, leaving a **61 ms** model allocation, not a 547-ms native bottleneck.
Two profiled sites carry most inflation: mHC pre-mix **388 ms** and grouped
Q8 vector work **157 ms** in gaps. Calling those native kernel costs would
send the next design in the wrong direction. Other GPU perturbations are
**-4.5% FN400, +0.3% FN2000, -0.7% GLM400**. GLM2000 category shares have
correspondingly weaker causal precision; the native totals and route census
remain ordinary measurements. Pure empty-launch models are only **9/24 ms
FN**, **5/14 ms GLM**. Host encoding/synchronization is a small target here.

## Width cliffs and record audit

| Family | 400 native ms | 512 native ms | 2000 native ms | 2048 native ms |
| --- | ---: | ---: | ---: | ---: |
| fn | 580.2 | 386.9 | 1522.4 | 1063.0 |
| glm | 1089.3 | 708.6 | 2341.8 | 2073.1 |

These are same-prefix shape controls, not equal-work speedups or candidates.
The larger aligned calls are faster despite processing more rows.

- **FN:** 400 misses the >=512 dense/expert/bucketing routes; 400/2000 both
  miss several 128-aligned HC/small-F32 gates, and IQ4_NL/Q2_0 down retains
  its 2048-only tensor admission. At 400, the largest diagnostic matrix
  completions are generic F32 **78 ms**, IQ4_NL down **60**, Q6 **58**, BF16
  **47**, IQ2_S gate/up **46** and IQ3_XXS gate/up **40**. At 2000, generic
  F32 is **232 ms**, IQ4_NL down's interval core **152**, Q6 **109**, BF16
  **107**, off-route Q5 **75**. Sparse attention remains **348 ms** of the
  model budget. GDN's 16-row internal tail exists, but its entire recurrence/
  glue allocation is smaller than the learned-kernel gap; it is not the lead.
- **GLM:** the later **November 7 signed-IQ widening is already on** for
  irregular >=1024 rows. Do not repeat it. At 400, Q6 generic **226 ms**,
  IQ2_S gate/up **354** and IQ3_S down **176** dominate. At 2000, IQ/Q6
  tensor paths engage, while the 512/1024/2048-only sparse attention gate
  retains F32 split attention: selected partials alone are **657 ms** in
  the diagnostic interval core. Tensor attention's extra query/probability
  Half rounding previously failed the no-worse F64 check (**569x RMS**);
  simply widening that guard is not an acceptable design.
- At 400, **79.6% FN / 78.2% GLM** of touched buckets have <=16 slots.
  Uniform N32 expert panels imply **3.02x / 2.83x** useful-slot padding;
  N16 would reduce that to **1.88x / 1.75x**. This is a padding-work predictor,
  not the same factor of wall time. Large-row adaptive panels already exist.
- Read record: November 5 tensor/GDN qualification; November 6 short-row,
  wide-Q6, F32-index and QSA selection/value work; November 7 narrow-F32,
  KDA/Q6 and signed-IQ row extensions; November 8 KDA shared-Half composition.
  The current gates are in `qwen4exp_prefill_tensor.rs`, `qwen4exp_moe.rs`,
  `glm5_next_metal/row_policy.rs` and `glm5_next_metal/prefill_attention.rs`.
  No old failed Half attention/down proposal is promoted by this report.

## Design swing and TTFT budget

Build **shape-complete native-quant prefill**, preserving each incumbent
node's operand rounding. For FN, cover actual M400/M2000 and HC alignment tails
with masked dense/HC tiles and count-adaptive N8/16/32 expert panels; decode a
weight slab once per panel, reuse it across the live slots, retain post-down
routing weights. Treat ragged internal GDN/QSA tails explicitly; padding whole
model timesteps would change state and is not this design. The 2048 IQ4_NL down
route is an implementation reference, not blanket numerical qualification.

For GLM, the small-row matrix package covers **128..511** without adding
operand narrowing. The long-row swing is **F32-query/F32-probability selected
QK/softmax/PV**, staging the original F16 latent cache to F32 tensor fragments.
Reuse the admitted attention partial backing for scores; preserve visibility,
selection, softmax precision and all 64K history. The existing F16 attention
route is a layout/scheduling reference, not the arithmetic to widen.

| Family | Projected warm unqueued TTFT contribution, 400 / 2000 | Savings versus measured native | Work / first kill |
| --- | --- | --- | --- |
| FN | **0.33–0.39 / 1.05–1.15 s** | **0.19–0.25 / 0.37–0.47 s** | About one week including qualification. First prove ragged F32/HC + native-quant panels beat the actual 400/2000 callers, not just an operator. |
| GLM | **0.55–0.70 / 1.9–2.0 s** | **0.39–0.54 / 0.34–0.44 s** | Small-row matrices 2–4 days; F32 sparse tensor path another 3–5 plus qualification. First kill F32 selected attention at actual 400/2000, with no-worse F64 error. |

These are **conditional engineering budgets**, not measured candidate TTFTs.
FN is anchored by the 387/1063-ms aligned controls; GLM400 by 709 ms at 512.
GLM2000 requires roughly 2x or better attention execution, targeting about
350–450 ms of its current 790-ms context allocation. Do not add independent
operator savings. These are the native PP contributions to a live-prefix TTFT;
serve tokenization, checkpoints, queueing and other agents' interference remain
outside the experiment and are not silently set to zero. Every saved native
PP millisecond returns GPU service to the concurrent decode scheduler.

Before defaults: frozen signed teacher-forced NLL/KL against a numerical-noise
control, no-worse F64 operators, structural/tail/state guards and whole-append
A/B. No quality conclusion follows from the measurement hashes. V4.1 is deferred:
its Engram host/I/O/overlap roof needs different instrumentation to be useful.

## Reproduction, retention and checks

Reusable scripts are under `scripts/research/append-roofline/`. Production
base **2bad80c4**, compiled diagnostic recipe **3023708c**; the committed
`prepare.py` installs all scratch seams and the example. Build **71 s** under
`lockf -k ~/.cache/qwen-llm/locks/build.lock nice -n 10 mise exec -- cargo build
--offline --release -p qwen-llm --example append_roofline`. Native commands,
model paths, source identity and thermal samples are in compact evidence under
`$HOME/qwen-runs/2026-11-09-append-roofline/` on both hosts. The
[CSV](2026-11-09-append-decomposition.csv) records every retained native sample
and derived roof/allocation input. The corpus recipe is in the
[steady packet](2026-11-09-steady-concurrency.md).

```sh
python3 scripts/research/append-roofline/prepare.py # isolated checkout only
python3 scripts/research/append-roofline/run.py --family fn \
  --binary target/release/examples/append_roofline --model "$MODEL" \
  --corpus "$CORPUS" --out "$RUNS/fn"
python3 scripts/research/append-roofline/analyze.py "$RUNS/fn/probe.jsonl" \
  --out "$RUNS/fn/summary.json"
# Repeat with --family glm and its artifact, in a fresh process.
```

Each model has one command per append, no drafting. Warm prefixes take
**42.65 s FN / 63.40 s GLM** (untimed). Nice 10, canonical Metal lease/wait,
25-second cooling + three zero readings, one-second monitoring. **No build
lock in timing**. Both controllers return zero, all sampled pressure **0**;
route-copy/profile/repeated native endpoints match bits. M5 owned worktree,
binaries/target and bundle removed **2026-11-09T06:31:55Z**. Full dispatch
bulk is deleted after aggregation; evidence is commit + command, no binary
archives. Six roof-accounting tests, seven existing steady-harness tests,
Python syntax, Rust formatting and docs/path/CSV checks pass. The M5 diagnostic
release build passes; the redundant queued laptop recheck was canceled after
that success. No full GPU/lib rollout suite is claimed for this research-only
report. No merge, client/server changes or owned worker remains.

# Performance Roadmap

Living cross-session performance plan for qwen-llm. Keep this file concise and
current: update the ranking when new measurements change expected value, risk,
or dependencies. Treat `docs/PLAN.md` as the architecture/history plan; this
file is the active optimization queue.

Working rule: optimize from causal performance hypotheses, not from measurement
novelty. Every measurement task in this file should exist only to kill or
confirm a concrete engine hypothesis.

For append-only checkpoint history and exact current handoff state, see
`docs/PERF-LOG.md`.

## Current North Star

Beat llama.cpp across dense and MoE Qwen 3.5/3.6 workloads, ideally by more
than a little, without taking shortcuts that fail at long context or larger
model shapes.

Primary guardrails:

- Dense: `Qwen3.6-27B-Q4_K_M.gguf`
- MoE A3B: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- MoE A10B: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`
- Never run performance benchmarks in parallel.
- Always keep dense 27B in perf analysis while optimizing MoE.

## Latest Baseline Snapshot

M4 Max, release `qwen-bench`, sequential runs.

| Model | Context | Total ms/token | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| 27B dense | 4K | 42.80 | 23.4 | after dense group6 NWG64 |
| 27B dense | 16K | 46.11 | 21.7 | attention ~14.3 ms, GDN/FFN dominates |
| 27B dense | 32K | 51.25 | 19.5 | attention ~19.3 ms |
| 35B A3B | 4K | 14.65 | 68.2 | NWG64 guardrail held |
| 35B A3B | 16K | 17.12 | 58.4 | NWG64 guardrail held |
| 35B A3B | 32K | 20.45 | 48.9 | NWG64 guardrail held |
| 122B A10B | 4K | 31.42 | 31.8 | group16 tile4 + NWG64 |
| 122B A10B | 16K | 33.47 | 29.9 | group16 tile4 + NWG64 |
| 122B A10B | 32K | 35.14 | 28.5 | group16 tile4 + NWG64 |

Prompt-only anchors, release `qwen-bench pp`, synthetic prompts:

- `qwen-llm` 9B dense packed pp: `~711.8 t/s`; `llama-bench`: `~824.0 t/s`
- `qwen-llm` 27B dense packed pp: `~212.0 t/s`; `llama-bench`: `~240.9 t/s`
- `qwen-llm` 35B A3B grouped MoE default now allowlists fused route+bucket plus
  hot-expert `n32` and the `E8xP32` router-logits kernel when `chunk_p >= 512`:
  about `690 t/s` at `pp256`, `798 t/s` at `pp512`, and `815 t/s` at `pp1024`;
  `llama-bench pp320` anchor is still `~1222.4 t/s`
- `qwen-llm` 122B A10B grouped MoE default now allowlists the same combo when
  `chunk_p >= 512`: about `264 t/s` at `pp256`, `316 t/s` at `pp512`, and
  `325 t/s` at `pp1024`; `llama-bench pp320` anchor is still `~393.3 t/s`
- Experimental `QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL=1` branch on the same
  exact grouped backend currently measures about `816 t/s` on 35B A3B `pp512`
  and `335 t/s` on warmed 122B A10B `pp512`, but only small additional gains by
  `pp1024` (`~822 t/s`, `~333 t/s`). Treat it as a likely keeper branch for the
  `pp512` regime, not as proof that the main MoE prompt gap is solved.
- prior repeated-prompt `qwen-llm` 27B dense packed prefill: `~205.4-205.9 t/s`
- current `llama.cpp` bounded `llama-cli -st` baseline: `~206.7 t/s` prompt,
  `~22.5 t/s` generation

Recent confirmed wins:

- Grouped MoE routed prefill had a real correctness bug: grouped `Q4_K` SwiGLU
  used `u32::MAX` as an open-ended expert-count sentinel while the Metal kernel
  cast it to signed `int`, turning the bound into `-1` and early-returning active
  experts. After fixing the sentinel, A10B smoke / boundary and A3B
  prefill-vs-single gates recovered, the fused route+bucket oracle became clean,
  and the routed MoE prompt story had to be re-based.
- That corrected re-baseline changes the active MoE path ranking:
  - `route_bucket` itself is small,
  - routed `grouped_swiglu` is still the largest MoE prompt bucket,
  - router logits are the next meaningful routed cost.
- The first post-fix routed-compute attack that converts end-to-end is a GPU-owned
  hot-expert grouped-Q4 split over the existing per-expert `counts/ids` ledger.
  Combined with fused route+bucket and allowlisted at `chunk_p >= 512`, it moves
  A10B from about `297.0 -> 302.9 t/s` at `pp512` and `299.6 -> 309.3 t/s` at
  `pp1024`, and A3B from about `736.3 -> 744.9 t/s` at `pp512` and
  `754.7 -> 776.7 t/s` at `pp1024`.
- A router-only `F32` `E8xP32` kernel now removes most of the remaining router
  logits cost on the proven MoE prompt shapes. Corrected A10B `pp512`
  `route_logits` falls from `5.95 ms` to `0.47 ms`, and A10B `pp1024`
  `route_logits` is `1.03 ms`. Allowlisted with the existing composed MoE prompt
  path at `chunk_p >= 512`, it moves A10B from about `302.9 -> 315.5 t/s` at
  `pp512` and `309.3 -> 324.7 t/s` at `pp1024`, and A3B from about
  `744.9 -> 797.6 t/s` at `pp512` and `776.7 -> 815.1 t/s` at `pp1024`.
- The clean post-`v0.100` family sweep materially updates the MoE prompt picture:
  A10B is now about `0.72x` at `pp512` and `0.76x` at `pp1024`, while A3B is
  still much farther behind at about `0.55x` / `0.58x`. That keeps MoE prefill as
  the main scoreboard gap even though the current exact grouped path is much
  better than the old baseline.
- Post-`v0.100` exact local grouped-MoE search-space elimination is now large:
  `F16` grouped inner default, cold `n8`, hot `th32` rollout, hot grouped-down
  atomic accumulate, persistent/locality queues, hot `32x32`, active hot tile
  lists, and a resident paired gate/up mirror all failed to produce a default-
  worthy win on this hardware / repo shape. Treat the exact local
  `grouped_swiglu` variant family as plateaued until a new mechanism is found.
- Grouped routed `inner/out` zero-fill is now correctness-covered as a real but
  small cleanup lever (`~0.5-0.8%` on the current prompt guardrails), not a main
  roadmap item.
- A bounded exact MoE prompt concurrency branch is now real: overlapping the live
  grouped routed tail with the live shared FFN at `chunk_p >= 512` preserves the
  current correctness matrix and converts to about `1-3%` end-to-end on `pp512`,
  but fades by `pp1024`. This is the strongest near-term exact production branch,
  not the main structural MoE answer.

- MoE decode now has a real GDN-side concurrency win. Reusing the dense
  concurrent-GDN front-projection split inside MoE decode and making it the repo
  default with `QWEN_DECODE_MOE_CONCURRENT_GDN=0` as a rollback path moves A3B
  from `~74.0/73.6 t/s` to `~77.8/78.0 t/s` at `tg32/tg128`, and A10B from
  `~32.4/32.5 t/s` to `~35.1/34.9 t/s`. At `ctx=4096`, decode-window also moves
  A3B `66.6 -> 71.5 t/s` and A10B `31.4 -> 34.0 t/s`, with GPU time improving on
  every measured row.
- A10B `pp128` cold variance is now explained as expert-bank first-touch, not as
  a steady-state runtime miss. Bench-only `QWEN_PP_WARM_MOE_BANKS=1` and
  `QWEN_PP_RESIDENCY_SET=1` collapse the cold outliers while leaving steady-state
  `pp320/pp512` flat, so treat them as benchmark methodology knobs rather than a
  top runtime roadmap item.

- `qwen-bench pp` now exposes a phase-matched prompt-only harness and lowering
  summary for dense/MoE prompt work. It confirmed the dense `pp320` miss is real
  (`~710/824 t/s` on 9B, `~211.9/240.9 t/s` on 27B) and uncovered the largest
  MoE issue: Q8 mixer projections were falling through to decode-shaped
  per-token paths.
- Enabling `Q8_0` packed mat-mat eligibility for MoE GDN/attention projections
  moves A3B pp320 from `~96 t/s` to `~194-198 t/s` and A10B pp320 from
  `~37.6 t/s` to `~85.1 t/s`, while dense 9B/27B guardrails stay flat.
- Activating the packed routed expert tail after fixing its production dtype gate
  and Q4_K byte layout moves A3B pp320 to `~261.5 t/s` and A10B pp320 to
  `~106.8 t/s`. The win holds across prompt lengths: A3B `pp64..pp1024` stays
  `~249-261 t/s` vs fallback `~188-197 t/s`; A10B `pp128..pp512` stays
  `~105-107 t/s` vs fallback `~84-85 t/s`.
- Batching the shared-expert branch plus rowwise shared gate/residual is another
  major MoE prompt win: A3B pp320 moves to `~399.8 t/s` and A10B pp320 to
  `~148.9 t/s`; default chunk-128 pp320 is `~375.9 t/s` on A3B and
  `~150.4 t/s` on A10B. Dense 9B/27B guardrails remain flat.
- A fully GPU-owned grouped expert-major routed backend now lands as the current
  production-resolution MoE prompt path for the proven `Q4_K/Q4_K/Q5_K`
  envelope. Default chunk-128 pp320 rises to `~556.4 t/s` on A3B and
  `~218.1 t/s` on A10B; chunk320 reaches `~710.4 t/s` / `~278.7 t/s`. Dense
  9B/27B guardrails stay flat.
- Packed attention-body cleanup now batches consecutive-position RoPE for Q/K and
  scatters the whole chunk's K/V rows into the cache in one dispatch before the
  per-token attention loop. On the repeated 320-token 27B prompt, packed prefill
  moves from ~201.5-202.3 t/s to `~205.4-205.9 t/s` and trims total wall from
  ~1582-1588 ms to ~1554-1558 ms.
- Fresh local `llama-cli -st` checks now show dense user-facing parity is real:
  `llama.cpp` is about `206.7 t/s` prompt / `22.5 t/s` generation on the same
  repeated prompt, while `qwen-llm` is already `~205.6 t/s` on prompt-only runs.
- Fresh local `llama-bench` still says the harder pure-prompt target is much
  higher: `pp320 ~240.9 t/s`. That keeps prompt-only parity, not CLI parity, as
  the main scoreboard target.
- Packed dense GDN prep is now over the whole prompt chunk. The old packed GDN
  prep loop used `P` launches of `ssm_conv_silu`, two L2 norms, and three
  scatters before the packed recurrence; the new path replaces that with one
  packed prep kernel plus two in-place batched L2 norms. On the repeated
  320-token 27B prompt, packed prefill moves from ~186.3 t/s to
  `~201.5-202.3 t/s` and trims total wall from ~1718 ms to ~1582-1588 ms.
- The new `QWEN_PREFILL_GDN_SPLIT` diagnostic confirmed the old packed GDN prep
  loop was the dominant GDN sub-bucket before the rewrite: about `~146 ms` wall
  / `~131 ms` GPU on the repeated prompt, versus only `~44 ms` for the packed
  recurrence itself.
- Production-shape packed-prefill profiling is now live in `qwen-bench decode`:
  prompt-only runs report total prefill GPU ms, and profiling no-op flags can
  remove dense FFN, GDN body, or attention body inside the real packed prefill
  graph.
- Packed dense GDN `rmsnorm_gated` is now batched over the whole prompt chunk
  instead of one dispatch per token. On the repeated 320-token 27B prompt, this
  moves packed prefill from ~183.1 t/s to ~186.2-186.5 t/s and trims GPU total
  from ~1717 ms to ~1690 ms.
- Latest same-prompt dense read is now `~205.6 t/s` on 27B, which cuts the fresh
  `llama-cli -st` prompt gap down to roughly three percent even though the harder
  `llama-bench` pure-prompt gap still remains.
- Production-shape dense prompt no-op profiling says the remaining packed-prefill
  cost is real GPU work. After the packed attention-body cleanup, the old
  attention-body no-op delta falls from about `~162 ms` wall / `~156 ms` GPU to
  roughly `~120 ms` wall / `~118 ms` GPU, leaving the remaining GDN tail /
  out-proj path as the clearest non-FFN prompt target. That keeps isolated FFN
  mat-mat kernels exonerated as the hidden prompt mystery.
- Prompt-prefill scratch now skips the unused `[P, V]` logits pack on no-spec
  prompt paths, removing a large dead allocation from timed prefill and nudging
  dense 27B prompt throughput to ~173.3 t/s.
- Packed dense `gdn_step_decay` over prompt tokens is now live and materially
  improves dense prompt processing: the same-prompt 27B plateau rises from
  ~165.0 t/s to ~172.9 t/s while decode stays unchanged.
- Dense packed GDN `alpha/beta` batching was a major prompt win: same-prompt 27B
  prefill rose from ~141.9 t/s plateau to ~165.0 t/s plateau, with decode
  unchanged.
- Dense packed-prefill GDN-tail attribution is now in place and points at the
  true `step_decay` recurrence as the sharpest next dense tail target.
- Dense packed prefill chunk tuning was a major win: on the same 321-token 27B
  prompt, moving from inherited `P=16` to dense default `P=256` improved prompt
  throughput from ~77.9 t/s to ~140.7 t/s with decode unchanged; one-chunk
  saturation is ~141.9 t/s once `P >= 321`.
- Dense packed prefill is now the default no-spec path in `qwen-bench decode`
  for dense models; on a 321-token 27B prompt it improved prefill from 24.3 t/s
  to 78.1 t/s (~3.22x) with decode unchanged.
- MoE packed prefill stage 1 is now live in `qwen-bench decode`; on a 321-token
  prompt it improves prefill from 74.8 -> 91.3 t/s on 35B A3B and 32.1 -> 36.6
  t/s on 122B A10B, with decode essentially unchanged.
- MoE packed prefill chunk tuning also matters: tuned default `P=128` lifts the
  same prompt to ~95.3 t/s on 35B A3B and ~37.6 t/s on 122B A10B.
- No-spec GPU argmax decode path landed. Dense decode is neutral within noise;
  MoE decode improves modestly by avoiding full logits readback (~1.1-1.5% on
  A3B / 122B in current 64-token runs).
- Dense KV-Q8 prototype is currently a negative result on M4 for the existing
  v4 main-kernel structure; attention gets slower, not faster.
- Attention v4 now supports `group=4`, unlocking the small dense family
  (0.8B / 2B / 4B / 9B) as real long-context canaries instead of failing back to
  the old threadgroup-memory-limited attention path. The local 9B sweep now runs
  cleanly through 32K: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.
- Attach-mode decode tracing is now practical via `qwen-bench decode-window`, and
  `scripts/profile/trace-metal.py` gives a compact Metal timeline summary
  without hand-written one-off parsers.
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, and `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

Recent measured negatives:

- Generic grouped expert-major MoE routed FFN via CPU ledger + gather/scatter +
  generic per-expert mat-mat is strongly negative on both A3B and 122B.
- Re-based “split routed FFN sidecar” experiments now show that beating the old
  packed-slot path was the wrong denominator: against the live grouped backend,
  separate gate/up grouped matmats plus the existing `silu_mul` + grouped down
  only reach parity to slight loss on `pp512`, so broad split-sidecar work is
  demoted until a smaller `MUL_MAT_ID`-style proof beats the current grouped
  projection / routed tail directly.
- Shared-expert batched stage-2 rewrite is semantically correct but slower
  end-to-end on A3B packed prefill.
- F16 routed-inner traffic reduction on the live Q5-down MoE path is a wash to
  slight loser end-to-end.
- Dense paired `gate+up` prompt fusion is exact-correct but only `~1.04x` in the
  exact-shape 27B microbench at `N=321`, below the go gate.
- Forcing single Q4 prompt mat-mat to `NR1=16` is worse than the current `NR1=32`
  path at `N=321`, so easy tile narrowing is not the answer.

## Force-Ranked Next Bets

### 1. Hypothesis: part of the remaining `pp320` gap is harness/phase mismatch

Optimizes: decision quality and scoreboard fidelity.

Why it moves to the top:

- We are near parity on the user-facing CLI harness but still far behind on
  `llama-bench pp320`.
- Before another major rewrite, we need to know how much of that miss is real
  prompt work and how much is benchmark semantics.
- This is the one measurement-heavy item that remains justified because it tests
  a direct causal hypothesis about the gap.
- The `MTL,BLAS` backend string `llama-bench` prints is a registration artifact,
  not a hot-path signal: at `pp320` on 27B Q4_K_M every mat-mat / mat-vec node
  runs on Metal and the BLAS backend executes zero ops. See the BLAS hot-path
  audit in `docs/PERF-LOG.md` for the per-node sched-debug evidence. The gap is
  Metal vs Metal, not "we are missing a CPU sgemm lane".

Current design rule:

- `qwen-bench pp` now covers the core harness need: synthetic `pp<N>`, no decode
  loop, optional tail skip, wall/GPU reporting, and lowering summaries.
- Keep using it as the scoreboard harness for dense and MoE prompt work; do not
  overfit to the older repeated-prompt decode harness.

Acceptance gates:

- Maintain `qwen-bench pp` while adding any future prompt phase buckets.
- Recompute the remaining prompt gap against `llama-bench pp320` after each major
  prompt-path change.

### 2. Hypothesis: if the `pp320` gap is real, decode-shaped prompt attention is the dominant remaining engine problem

Optimizes: the largest likely remaining structural prompt-only gap once harness
semantics are aligned.

Why it moves up:

- The cheap packed attention-body cleanup already landed, but if the pure prompt
  gap survives harness matching, GDN-tail cleanup alone cannot close it.
- Our prompt attention still retains decode-shaped structure in the inner loop.
- That makes prompt-native packed attention the strongest causal explanation for
  a large remaining `pp320` miss.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate after item 1 confirms the remaining miss is real and not mostly
  harness semantics.
- Prefer prompt-native packed attention / verify primitives over more small glue
  cleanups once the hypothesis survives.

Acceptance gates:

- End-to-end pure prompt throughput must move materially against the new harness.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 3. Hypothesis: if prompt attention is not enough, the next real dense prompt miss is GDN out-proj / recurrence tail

Optimizes: the residual dense prompt GDN work after the packed prep rewrite and
attention-body cleanup.

Why it moves back to the top:

- The split ladder still says the remaining GDN tail is mostly out-proj plus a
  smaller recurrence cost.
- This is now a bounded fallback hypothesis, not the assumed main story.

Current design rule:

- Keep using the real-graph `QWEN_PREFILL_GDN_SPLIT` ladder for bounded probes.
- Prefer narrow out-proj / recurrence cleanups over another broad rewrite.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 4. Dense Prompt: Attention Body Cleanup

Optimizes: any additional dense prompt throughput still available in the
full-attention layers.

Why it stays near the top:

- The packed consecutive RoPE + chunk-scatter cleanup was a real win, but the
  attention body still costs on the order of `~120 ms` wall on the repeated
  prompt.
- If the remaining GDN-tail work stalls, this remains the best smaller-bore
  alternate lane before a true packed prompt attention rewrite.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate to more invasive packed causal prefill attention if smaller
  cleanups stop moving the prompt.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 5. Decode Command-Model Overlap

Optimizes: apples-to-apples dense decode latency, especially 27B at 4K and up.

Why it moves up:

- Real 27B 4K attach-mode Metal trace now exists.
- It shows `128` command buffers for `128` decode tokens, `128` compute encoders,
  encoder duration median `~0.699 ms`, and previous completion -> next submit
  median `~0.538 ms`.
- The direct decode-window profiler at 4K is the decisive result:
  `med_total ~42.69 ms`, `med_gpu ~42.14 ms`, `med_cpu_enc ~0.20 ms`, so decode
  is about `98.7%` GPU-busy on the 27B dense guardrail at 4K.
- The token loop is fully serialized today. The trace does NOT show a giant
  hidden bubble, but it does show a real low-single-digit command-model gap.
- Process-scoped compute intervals split into a small short-gap population and a
  large token-cadence population; the short intra-CB gaps total only about
  `~1.2 ms/token` at 4K, which keeps this as a real but bounded lever.

Immediate focus:

- Double-buffered / pipelined decode submission first.
- Use the new attach-mode trace helper and parser to validate any overlap claim.
- Only escalate to heavier encoder restructuring if post-overlap traces still
  show meaningful serialized slack.

Acceptance gates:

- Reduce completion -> next-submit gap and total 27B decode ms/token at 4K.
- Keep exact-token behavior and current correctness gates intact.

Status:

- A dense-only bench path now exists at `qwen-bench decode-window --pipelined`.
- Measured at 27B dense:
  - `ctx=4096`: about `~0.3%` over alternating repeats
  - `ctx=32768`: about `~0.3%`
- Keep it as an experimental harness, not a production checkpoint, unless a
  future shape/context shows a materially larger win.
- A second bench-only dense decode branch now exists at
  `qwen-bench ctx-sweep --concurrent-gdn-proj`.
- At 27B dense `ctx=4096`, `window=64`, it improves decode from
  `43.87 -> 42.14 ms/token` (`22.8 -> 23.7 t/s`), with the gain showing up in
  GPU time rather than CPU encode.
- At 27B dense `ctx=16384`, `window=64`, it also improves decode from
  `47.67 -> 46.51 ms/token` (`21.0 -> 21.5 t/s`).
- This is the first command-model branch that has cleared the “real enough to
  checkpoint” bar; the gain survives 4K and 16K, though it narrows somewhat as
  attention grows.
- Attention-only overlap is smaller:
  - `ctx=4096`: `43.89 -> 43.33 ms/token`
  - `ctx=16384`: effectively flat (`47.26 -> 47.22 ms/token`)
- Running both projection-overlap branches together is still positive and
  checkpoint-worthy:
  - `ctx=4096`: `43.64 -> 42.31 ms/token`
  - `ctx=16384`: `47.23 -> 46.14 ms/token`
- The combined branch is not additive with GDN-only overlap, but it remains the
  strongest decode-focused command-model variant measured so far.
- Treat encoder-boundary removal in these concurrent paths as an anti-bet until
  a fresh A/B proves otherwise: the split is buying GPU-side overlap between
  independent front projections, not merely adding host encode overhead. Host
  encode is already only ~0.20 ms of ~42.14 ms GPU time at 27B dense 4K, so the
  upper bound on collapsing encoders is well under 0.5%; the load-bearing piece
  is the middle `begin_concurrent` encoder for the front projections, and any
  merge must preserve that concurrent-dispatch property.
- MoE no longer shares the same decode-overlap uncertainty: concurrent GDN front
  projections are now production-wired for MoE decode on this repo, and the A3B /
  A10B `tg32/tg128` sweeps are already materially positive.

### 6. Read-Only Weight Residency And Scratch Storage Cleanup

Optimizes: decode and prompt wall via cheaper Metal bookkeeping and cleaner GPU
memory behavior.

Why it belongs near the top now:

- The command-model trace says there is not a giant host bubble, so the cheap
  structural wins become more attractive than speculative scheduler work.
- The external review's `hazardTrackingMode: untracked` + residency-set idea is
  orthogonal to no-copy GGUF views and should help regardless of mmap strategy.
- Scratch is still `StorageModeShared` everywhere today, which is convenient but
  not obviously ideal for GPU-only hot tensors.
- The new A10B `pp128` cold-run study keeps this bounded: touching or pinning MoE
  expert-bank buffers fixes a benchmark-hotness artifact, but steady-state
  `pp320/pp512` stay flat, so this is still structural cleanup rather than the
  highest-EV runtime lever.

Immediate target order:

1. Mark read-only weights as untracked and managed by a residency set.
2. Audit CPU readback/debug use, then move only proven GPU-only scratch arenas
   toward `StorageModePrivate`; avoid a blanket allocator swap.
3. Measure decode/prefill again before bundling this with larger graph changes.

Acceptance gates:

- Any change must preserve correctness and avoid regressing steady-state decode.
- Keep these as cheap structural cleanup unless traces show a larger-than-expected
  wall effect.
- Specifically: `--full-logits-decode` (exact-token oracle) and any tap-based
  hidden-state captures must stay green across a scratch storage-mode change,
  since a blanket `StorageModePrivate` swap silently breaks CPU readback paths.
  `MetalSession::fresh` currently allocates ~39 scratch tensors via
  `MetalTensor::zeros_f32` (`StorageModeShared`); the audit must classify each
  as GPU-only vs CPU-readable before any allocator change lands.

### 7. MoE Next: Execution-Model Reset After The Exact Grouped Plateau

Optimizes: the remaining MoE prompt scoreboard gap after the current exact grouped
backend has locally plateaued on this hardware / repo shape.

Current read:

- The grouped expert-major MoE prompt backend is now the stable base path again.
- Fused route+bucket and hot-th48 grouped Q4 are both kept because they compose
  positively at `chunk_p >= 512`.
- `n32-all` is still not a ship candidate: it improves grouped compute locally but
  does not clear the end-to-end pp gate.
- Router logits no longer look like the next missing kernel. The `E8xP32` kernel
  is already good enough to ship on the proven regime.
- The obvious exact local `grouped_swiglu` variant family is now well sampled and
  mostly exhausted here: tile, threshold, queue/locality, atomic-down, and
  resident-mirror branches all went flat or negative.
- Two more exact reads now narrow the field further:
  - grouped `inner/out` zero-fill is safe to skip and slightly positive, but too
    small to change the scoreboard by itself;
  - a re-based split routed FFN proof only reaches parity to slight loss versus
    the live grouped backend, so “just make it llama-like by separating gate/up”
    is not enough.
- The strongest near-term exact branch is now the guarded concurrent-tail path:
  overlap the live grouped routed tail with the live shared FFN at `chunk_p >= 512`.
  It converts to roughly `1-3%` end-to-end on `pp512`, but collapses quickly by
  `pp1024`, so it should be treated as a bounded production win, not the final
  structural answer.
- That leaves the next serious structural MoE-first branch as a **narrow** kernel
  proof, not a broad sidecar rewrite: a true `MUL_MAT_ID`-style / id-aware
  projection microproof that beats the current grouped projection or routed tail
  directly on both A3B and A10B.

Acceptance gates:

- Keep `QWEN_PREFILL_MOE_GROUPED=0` as the kill switch while rollout evidence is
  still expanding.
- Keep the route-only and hot-only flags forceable, but do not default either one
  globally outside the allowlisted combo regime.
- Default-on only for the proven `Q4_K/Q4_K/Q5_K` MoE packed-prefill envelope;
  fallback stays live for unsupported dtypes/shapes and for prompt chunks below
  the allowlist.
- Keep the current correctness matrix green: A3B single-token-loop + hidden
  capture, A10B smoke, awkward chunk-boundary A10B (`T=129`, `P=128`), and dense
  9B/27B guardrails.
- Treat the concurrent-tail branch as shape-gated until a full active-shape
  prefill oracle exists for the covered `pp512+` regime; do not assume the large
  block-local overlap effect composes into a large end-to-end win.
- Do not spend another major engineering branch on a new exact local
  `grouped_swiglu` variant unless a new diagnostic shows a new mechanism beyond
  the already-falsified tile / threshold / queue / mirror family.
- Before writing a large new id-aware kernel, require a smaller microproof to beat
  the current grouped projection / routed tail directly on both A3B and A10B, not
  just the obsolete packed denominator.
- Any next MoE-first branch must beat the current allowlisted combo on both A3B
  and A10B at `pp512` / `pp1024`, not just improve a routed microprofile.

### 8. Use 9B As The Fast Dense Long-Context Canary

Optimizes: experiment throughput and long-context turnaround while preserving the
27B guardrail.

Why it is now active:

- `group=4` attention v4 is now enabled, which unlocks the whole small dense line
  (0.8B / 2B / 4B / 9B) for realistic long-context decode.
- Local 9B now reaches 32K cleanly and is dramatically faster to iterate on than
  27B: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.

Usage rule:

- Use 9B for fast falsification of long-context attention / dense prompt ideas.
- Keep 27B in the analysis loop before claiming a real win.

Acceptance gates:

- Long-context experiments should be reproducible first on 9B, then confirmed on
  27B before the roadmap moves.

### 9. No-Copy GGUF Views And Residency Warmup

Optimizes: TTFT, cold-start variance, load-time memory pressure, and possible VM
object overhead.

Why it enters the roadmap now:

- The `ds4` close read makes this the strongest non-kernel structural crib.
- Current `qwen-llm` still copies weights tensor-by-tensor into fresh shared
  buffers; `ds4` instead wraps a few large GGUF-backed no-copy Metal views and
  warms residency up front.

Why it is not above the current prompt work:

- This is more likely a load / first-token / memory-cleanliness lever than the
  next steady-state prompt-throughput unlock.
- It still looks high-EV enough to prototype once the current MoE and dense
  prompt branches have a stable checkpoint.

Acceptance gates:

- Prototype shows materially better load time, first measured token stability, or
  memory / VM-object behavior without regressing steady-state throughput.

### 10. Frontier Benchmark Harness With Snapshot / Restore

Optimizes: benchmark quality and long-context decision speed.

Why it matters:

- `ds4-bench`'s frontier measurement style is a better mental model for prompt vs
  decode frontiers than one blended tokens/sec number.
- This would sharpen long-context dense/MoE comparisons and future speculative
  work without changing model semantics.

Acceptance gates:

- Add exact frontier prompt/decode probes that can restore from snapshots and
  measure a fixed local window.
- Use it to compare qwen vs llama phase-for-phase, not on blended totals.

### 11. Speculative Path: Attack Repeated Long-Context Attention Cost

Optimizes: DFlash / MTP viability at realistic context lengths.

Priority rule:

- Keep this behind the current dense/MoE prompt push.
- When returning to speculative work, do not lead with policy/schedule tuning;
  lead with kernel work that removes repeated long-context attention cost.

What the latest analysis says:

- Current MTP shape is structurally weak at long context because lazy verify does
  not amortize enough base work and MTP has its own growing KV attention cost.
- Current DFlash is made safe by adaptive verify `N`, but not fast, because the
  drafter and target still pay too much long-context attention work.

Highest-EV speculative kernel targets:

1. Target packed-verify multi-query attention so consecutive verify queries share
   KV reads.
2. DFlash two-range attention reading ctx-cache and noise directly, without
   `k_full` / `v_full` materialization.
3. Retile the `N=16` mat-mat specializations only if verify/draft phase profiles
   show N16 mat-mat-heavy surfaces remain material after the attention fixes.
4. Adaptive draft compute width, not only adaptive verify width.

### 12. Mid-Graph Flush / Overlap Before ICB / MTL4

Optimizes: decode and prompt wall only if later traces show more cadence slack at
other contexts or shapes.

Why it stays behind the others:

- The new 27B 4K trace shows a real but modest command-model gap, not a giant one.
- Cheaper overlap and residency work comes before heavier command-graph surgery.

Acceptance gates:

- Only pursue after double-buffered decode and structural cleanup are measured.
- Require trace evidence of additional idle gap before escalating further.

### 13. KV-Q8 / Quantized KV Cache For Long Context

Optimizes: long-context decode, DFlash usefulness at long context, memory.

Current read:

- The first dense KV-Q8 prototype is a negative result on M4: despite exact
  append quantization and good output similarity, the current Q8 v4 main path
  makes attention slower than the tuned F16 path.
- Codex-wrap review says the likely cause is structural: scalar Q8 dequant/load
  overhead is overpowering stored-byte savings against an already-strong F16
  vectorized path.

Expected payoff: still potentially large in theory, but only if a materially
different reader structure wins. Do not spend more blind sweep time on the
current implementation.

Risks and constraints:

- Easy time sink.
- Needs a fundamentally better Q8 read path or a different compression format to
  be worth revisiting.

Acceptance gates:

- Revisit only with a concrete new kernel structure and a fast feedback plan.
- Cut again quickly if attention does not beat F16 at 32K or 64K.

### 14. Dense Decode Surgery And Small Decode Hygiene

Optimizes: dense decode throughput and measurement integrity.

Why it stays late:

- Dense decode is already competitive enough that prompt work dominates the
  scoreboard.
- Prior FFN mega-fusion had weak payoff and GDN recurrence semantics are
  correctness-sensitive.
- GPU argmax is already landed; dense gain is neutral within noise and MoE gain
  is modest but real.

Acceptance gates:

- Any decode surgery must be driven by a fresh dense phase profile identifying a
  specific waste pocket.
- First candidate if the profile supports it: GDN decode recurrence tiling across
  multiple `dv` rows to reduce launch/TG overhead and duplicated Q/K loads, but
  only if `gdn_step_decay` is materially above its unavoidable state R/W floor.
  The reducible component is the redundant `q_h`/`k_h` device-pointer reload
  across the `head_dim × n_v_heads` TG grid (currently ~384× per head per token
  for `head_dim=128, n_v/n_k=3`); the state R/W itself (~6 MiB/layer/token,
  ~288 MiB/token across 48 GDN layers on 27B) is irreducible without restructuring
  the recurrence. Budget the win at ~2-4% decode if k/q-bound; 0% if state-R/W-bound.
- Exact-token A/B paths (`--full-logits-decode`) and argmax regression tests stay
  green while decode work proceeds.

## Deprioritized For Now

- FFN mega-fusion as a first move: prior layer-major fusion produced too little
  gain for the complexity.
- Giant GDN recurrence rewrite: correctness risk is too high without a sharper
  measured target.
- Synthetic-only attention tuning: attention is much improved; further tuning
  should be driven by full phase/ctx sweeps.
- Vocab pruning or approximate lm_head shortcuts without exact-token gates.

## How To Update This File

When a session changes performance direction, update only the smallest relevant
section:

1. Add new measured baseline rows or replace stale ones.
2. Move ranked bets only when measurements change expected value or risk.
3. Record accepted wins in "Recent confirmed wins".
4. Record failed experiments in "Deprioritized" or in the relevant bet's risks.
5. Keep benchmark notes sequential and reproducible; do not mix parallel runs.
6. At each improved checkpoint, make the diff tell one optimization story:
   short `v0.xx:` subject, detailed wrapped body with measurement + validation.
7. When a win changes the broader lowering or bottleneck picture, update
   `docs/INFERENCE-GRAPH.md` alongside this file so the semantic map stays in
   sync with the engine and the current performance story.

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```

# Performance Roadmap

Living cross-session performance plan for qwen-llm. Keep this file concise and
current: update the ranking when new measurements change expected value, risk,
or dependencies. Treat `docs/PLAN.md` as the architecture/history plan; this
file is the active optimization queue.

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

Prompt-only anchor, same repeated 320-token prompt:

- `qwen-llm` 27B dense packed prefill: `~201.5-202.3 t/s`
- current `llama.cpp` baseline: `212.44 t/s`

Recent confirmed wins:

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
- Latest same-prompt dense read is now `~202 t/s` on 27B, which cuts the fresh
  `llama.cpp` prompt gap down to roughly five percent.
- Production-shape dense prompt no-op profiling says the remaining packed-prefill
  cost is real GPU work. After the packed GDN prep win, the next large non-FFN
  dense prompt buckets are now attention body at about `~162 ms` wall /
  `~156 ms` GPU and the remaining GDN body at about `~156 ms` wall. That keeps
  isolated FFN mat-mat kernels exonerated as the hidden prompt mystery.
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
- Shared-expert batched stage-2 rewrite is semantically correct but slower
  end-to-end on A3B packed prefill.
- F16 routed-inner traffic reduction on the live Q5-down MoE path is a wash to
  slight loser end-to-end.
- Dense paired `gate+up` prompt fusion is exact-correct but only `~1.04x` in the
  exact-shape 27B microbench at `N=321`, below the go gate.
- Forcing single Q4 prompt mat-mat to `NR1=16` is worse than the current `NR1=32`
  path at `N=321`, so easy tile narrowing is not the answer.

## Force-Ranked Next Bets

### 1. Dense Prompt: Attention Body Cleanup

Optimizes: dense prompt throughput on the full-attention layers after GDN.

Why it moves up:

- Production-shape no-op profiling after the packed GDN prep win says attention
  body now costs about `~162 ms` wall / `~156 ms` GPU on the repeated 320-token
  prompt, making it the largest remaining non-FFN dense prompt bucket.
- Full-attention prefill is still decode-shaped in the middle: per-token RoPE,
  KV append, and decode attention inside the chunk.
- The packed GDN prep win already captured the obvious prep-loop waste on the GDN
  side, so attention now deserves the first focused follow-up.

Current design rule:

- Start with bounded cleanup around the existing attention math: batched
  consecutive-position RoPE and chunk-wise KV scatter / glue removal.
- Defer truly packed causal prefill attention until these cheaper cleanups are
  falsified.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B without regressing decode.

### 2. Dense Prompt: Remaining GDN Body Cleanup

Optimizes: the residual dense prompt GDN work after the packed prep rewrite.

Why it stays high:

- The new split ladder says the packed GDN body still totals about `~156 ms`
  wall on the repeated prompt even after the prep rewrite.
- Most of the old prep waste is gone, so the remaining GDN cost is now smaller
  and sharper: roughly `~95 ms` in the out-proj tail and `~40-45 ms` in the
  packed recurrence on the original ladder.

Current design rule:

- Keep using the real-graph `QWEN_PREFILL_GDN_SPLIT` ladder for bounded probes.
- Prefer narrow recurrence or out-proj cleanups over another broad GDN rewrite.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 3. Decode Command-Model Overlap

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

### 4. Read-Only Weight Residency And Scratch Storage Cleanup

Optimizes: decode and prompt wall via cheaper Metal bookkeeping and cleaner GPU
memory behavior.

Why it belongs near the top now:

- The command-model trace says there is not a giant host bubble, so the cheap
  structural wins become more attractive than speculative scheduler work.
- The external review's `hazardTrackingMode: untracked` + residency-set idea is
  orthogonal to no-copy GGUF views and should help regardless of mmap strategy.
- Scratch is still `StorageModeShared` everywhere today, which is convenient but
  not obviously ideal for GPU-only hot tensors.

Immediate target order:

1. Mark read-only weights as untracked and managed by a residency set.
2. Move GPU-only scratch arenas toward `StorageModePrivate` where the CPU never
   reads them.
3. Measure decode/prefill again before bundling this with larger graph changes.

Acceptance gates:

- Any change must preserve correctness and avoid regressing steady-state decode.
- Keep these as cheap structural cleanup unless traces show a larger-than-expected
  wall effect.

### 5. MoE Next: Token-Major Cleanup Or Custom Persistent Kernel Research

Optimizes: MoE prompt throughput on A3B / 122B after the generic grouped path was
falsified.

Why it moves down:

- The obvious grouped-expert version is now a measured negative result.
- Shared-stage batching and F16 routed-inner traffic reduction also failed to
  beat the current packed stage-1 MoE path end-to-end.
- The remaining MoE upside likely requires either smaller token-major cleanup or a
  genuinely custom persistent grouped kernel, not another generic gather/scatter
  experiment.

Acceptance gates:

- Any new MoE branch must explain why it avoids the generic grouped-GEMM failure
  mode before it gets implementation time.
- Keep MoE correctness gates and packed-MoE tail attribution in the loop.

### 6. Use 9B As The Fast Dense Long-Context Canary

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

### 7. No-Copy GGUF Views And Residency Warmup

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

### 8. Frontier Benchmark Harness With Snapshot / Restore

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

### 9. Speculative Path: Attack Repeated Long-Context Attention Cost

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
3. Adaptive draft compute width, not only adaptive verify width.

### 10. Mid-Graph Flush / Overlap Before ICB / MTL4

Optimizes: decode and prompt wall only if later traces show more cadence slack at
other contexts or shapes.

Why it stays behind the others:

- The new 27B 4K trace shows a real but modest command-model gap, not a giant one.
- Cheaper overlap and residency work comes before heavier command-graph surgery.

Acceptance gates:

- Only pursue after double-buffered decode and structural cleanup are measured.
- Require trace evidence of additional idle gap before escalating further.

### 11. KV-Q8 / Quantized KV Cache For Long Context

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

### 12. Dense Decode Surgery And Small Decode Hygiene

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

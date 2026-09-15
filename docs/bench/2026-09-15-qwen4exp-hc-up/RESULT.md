# HC Component Useful; Native Qualification On Hold

**Current follow-up:** the bounded packed-signature repairs now allow the unchanged
native HC packet to pass strict32-token/state checks and incremental timing gates.
See `NATIVE-RESULT.md` for current authority. The earlier HOLD/failures below remain
historical evidence, not the current first action.

Research implementation/reproducer: `60e644ae`. Isolated product MoE builtin
repair: `95a71437`. Local M4 Max, existing UD-Q3_K_XL; no new weight downloads.
All GPU runs hold the production benchmark lease and real wired-memory check.
Metal API validation remains enabled. No shader-validation claim.

## Decision And Leverage Map

Keep the research-only rank-320 HC up-plus-mix candidate and the independently
qualified grouped-IQ3 builtin-width repair. Do not deliver an HC option or change
HC production routing. Native HC has **no numerical or timing result**: both
attempts abort during the unchanged packed prefix, before HC candidate execution.

1. Localize the remaining packed-prefix API-validation boundary; its kernel is
   not yet identified. Do not assume every narrow builtin requires modification.
2. Qualify the smallest isolated repair, then resume the unchanged native HC
   packet with split-QSA enabled in both arms. No fresh numerical/performance gates.
3. Obtain a bounded complete-MoE cost observation before another expert topology.
4. Keep HC body sweeps, attention-body polishing, and RMS broadcast parked.

The complete-HC component result earns native investigation, not native attribution
or request-throughput claims. The prior provisional GDN-containing block total
includes HC/FFN and is not recurrence cost.

## Component Result

Twelve independent full-shape Q8 down/up sets total83,558,400 bytes; up-only
41,779,200 bytes. Four branches, hidden2560, rank320. The candidate replaces
5120 generic up-projection threadgroups plus mean with640 four-SIMDgroup groups.
It reuses the existing low SiLU and writes raw gates for trace compatibility.

Protocol 1 **FAIL**,0.30s, before timing: the incumbent violates the original
pointwise raw-dot threshold on x256 low inputs with strong cancellation. All
twelve ordinary fixtures and zero/tiny/alternating-sign cases pass both arms.
The candidate's x256 case was not reached in that first attempt.

One diagnostic-only capture,0.05s, retains the same inputs and both full outputs:
original raw pointwise failures12/18, raw RMS8.7734e-8/1.2565e-7, zero power-of-two
scaling discrepancies, zero original mixed pointwise failures. Mixed RMS is
9.9284e-7/1.0804e-6. The diagnostic supports cancellation sensitivity in the
result-only bound, not candidate superiority. Its m20 incumbent estimate assumes
a SIMD reduction tree and is not a certified compiled-kernel error bound.

Protocol 2 was explicitly frozen before timing: only the x256 raw pointwise
criterion becomes a common conditioning-aware gamma45*S bound for both arms,
with F64 reference allowance. Original raw RMS, all mixed-output gates, and all
other pointwise gates remain unchanged. Original failures are still reported.
See `PROTOCOL-2.md`; protocol1 is not retrospectively promoted.

Protocol-2 numerical screen **PASS**, including guarded offsets, poisoned outputs,
immutable weights/inputs, upstream and timed-replay bitwise checks. Ordinary
candidate raw maximum absolute error3.9266e-6, mixed2.3609e-7. On scaled input,
candidate raw maxabs0.000754338/RMS1.2565e-7 and mixed maxabs1.1324e-5/RMS1.0804e-6.

One fixed warm ABBA then measured ABBA, eight repeats per command. Milliseconds
below are normalized to one twelve-read chain; all readbacks outside clocks.

| Chain / metric | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Up + mean GPU | 0.581844 | 0.234109 | 0.225932 | 0.569641 |
| Up + mean encode/submit/wait wall | 0.653859 | 0.293792 | 0.284583 | 0.643635 |
| Complete HC read GPU | 0.818250 | 0.505307 | 0.515734 | 0.835531 |
| Complete HC read encode/submit/wait wall | 0.979625 | 0.715755 | 0.733818 | 1.047516 |

Complete-read GPU saving **38.2602%**, control spread **2.0899%**. Mean and both
pairs clear the frozen>=10% and extrapolated>=0.5ms/97-read floors. Extrapolated
saving2.5573ms is NOT a native measurement. Leaf saving60.0479% is diagnostic;
it is not substituted for the complete-read gate. Complete read excludes HC
injection and downstream mixer/FFN. Test wall0.37s; no width sweep or timing retry.

## Validation Boundary And Isolated Repair

Native attempt01 aborts with `65536 must be <=65535 for tiitg` during packed
prefill. A single trace localizes the offending dispatch to grouped IQ3_XXS
SwiGLU N16, grid128x10x512, TG128x1x1. The source local index is0..127; the
validator/compiler's internal derivation of65536 is not established here.

The bounded model-free packet, all under API validation:

- Original narrow builtin, depth511: PASS0.11s.
- Original narrow builtin, depth512: expected assertion reproduced.
- Only annotated builtin widened to uint, immediately narrowed to original local
  ushort, depth512: PASS0.10s. Same grid/bindings/dynamic shared memory.
- Nonempty original/widened comparison: bitwise PASS0.03s,26 active permuted
  slots,1664 nonzero outputs,17-token N16 tail, guarded buffers/unused slots.

`95a71437` keeps only the minimal product signature repair and removes the paired
shader/temporary trace. Three retained product regression tests pass0.13s at
depth511/512 and nonempty same-kernel bitwise replay. Historical narrow/wide
equivalence belongs to `60e644ae`, not the retained replay test.

Native attempt02 still aborts before prefix completion, naming `tiitg` rather
than the repaired parameter `tiitg_wide`. This is consistent with another kernel
but does not identify it or prove how far the graph progressed. Per the frozen
stop rule, no further model replay, broad builtin edits, or validation bypass.
The repair is **validated for the isolated grouped-IQ3 boundary; full packed-prefix
validation remains unresolved**. Native logit artifact directories are empty;
there are no native HC rows/state/performance results to promote.

CPU scoped-probe nesting/unwind test passes. Final `cargo check -p qwen-cli --bin
qwen` passes without warnings. Independent read-only review supports retaining
the isolated repair and research candidate while parking native qualification.
Reviewer: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.

## Evidence And Reproduction

Raw logs in the optimization worktree's `target/profiles/`:

- `2026-09-15-qwen4exp-hc-up-screen-01.log`: original failure, no timing.
- `2026-09-15-qwen4exp-hc-conditioning-01.log` and
  `qwen4exp-hc-conditioning-94435/`: full raw/mixed/reference/S captures and JSON.
- `2026-09-15-qwen4exp-hc-up-screen-v2-01.log`: useful component result.
- `2026-09-15-qwen4exp-native-hc-up-01.log`,
  `2026-09-15-qwen4exp-native-hc-validation-trace.log`, and
  `2026-09-15-qwen4exp-native-hc-up-02.log`: blocked native attempts/localization.
- `2026-09-15-qwen4exp-moe-index-{narrow-511,narrow-512,wide-512,populated}.log`:
  historical isolated packet; `2026-09-15-qwen4exp-moe-index-product-regression.log`:
  final retained regressions.

Run individual GPU tests serially, never the whole ignored suite. The protocol-1
test intentionally retains its old failing criterion. The current component test:

```sh
MTL_DEBUG_LAYER=1 cargo test --release -p qwen-llm --lib hc_up_k320_screen_v2 -- \
  --ignored --exact qwen4exp_metal::hc_up_screen::hc_up_k320_screen_v2 \
  --nocapture --test-threads=1
```

Do not repeat the native packet until the remaining prefix boundary is isolated.
The previously approved server shutdown remains in effect; this work does not
restart it or stop any additional server. No remote push.

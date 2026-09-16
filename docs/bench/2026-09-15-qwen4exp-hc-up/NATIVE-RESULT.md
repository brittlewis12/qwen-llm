# Native HC Qualification PASS

Historical research qualification. Later actual-product numerical/CLI checks
pass but performance promotion is HOLD; current authority is `PRODUCT.md`.

Current authority supersedes the earlier native HOLD, without changing the
historical failures in `RESULT.md`. HC remains research-only: no production HC
option or default change. Existing UD-Q3_K_XL, local M4 Max, production GPU lease
and real wired-memory check held throughout, Metal API validation enabled.

## Packed Validation Repairs

The remaining trace identifies IQ4_NL down M64xN32 at grid64x40x512/TG128. Paired
reproducer `958adb2f` demonstrates narrow511 PASS, narrow512 assertion, wide512
PASS, and bitwise populated equivalence across N32/M64 tails. Product fix
`82a21f9d` changes only the annotated builtin to uint, immediately narrowing to
the original ushort local. Three product regressions pass in0.12s, including
the checked host route matching the original narrow output fingerprint.

Native03 still aborts before prefix completion. To avoid repeated model runs,
a CPU-only metadata inventory bounds the other reachable packed-MoE signatures:

| Operation / dtype | Layers | Status |
|---|---|---|
| Gate/up IQ3_XXS | 47 layers | Previously repaired at `95a71437` |
| Gate/up IQ4_XS | Layer 2 | Independently reproduced, then repaired |
| Down IQ4_NL | 43 layers | Repaired at `82a21f9d` |
| Down Q8_0 | Layers 2,4,30,46,47 | Independently reproduced, then repaired |

All expert banks have512 experts. The IQ4_NL M128xN16 option is restricted to
512/527-token chunks; this packet uses2048/3/128 and does not justify editing that
variant. The inventory is source/metadata evidence, not a blanket runtime claim.

Both remaining narrow signatures independently reproduce the512-expert assertion.
The first empty Q8 probe had an unused nb01 unit error (20 instead of680 bytes);
it is retained as signature-only evidence. The corrected exact-argument rerun
also reproduces the assertion before any weight access. No populated test used
the wrong stride.

Paired packet `c580f0a0`: six tests PASS0.26s, including original511 controls,
widened512 cases, and narrow/wide populated equivalence for both signatures.
Each populated case has42 active/24 untouched slots, both tile tails, canaries,
immutable inputs,2940 nonzero outputs, and full captures saved before assertions.
Product fix `09f54356` removes the paired variants and preserves local arithmetic.
Six retained regressions PASS0.19s; checked production host outputs match original
narrow captures, not merely a same-kernel replay.

No unrelated signatures or dispatch shapes changed. This is validation
compatibility plus bounded numerical equivalence, not a MoE speedup claim.

## Native Numerical Qualification

The unchanged `NATIVE-PROTOCOL.md` packet now completes in32.74s. One current
packed SSH2179 prefix,32 teacher-forced continuation tokens, no second reference
prefill or new weights. Product split-QSA is ON in both arms, so HC improvement
is incremental to the already delivered attention option.

Baseline/restored-baseline reproduces all32 rows, hyper residuals, and persistent
state bitwise. Candidate census witnesses3104 HC calls and384 split-QSA calls,
with no incumbent singleton HC mean. Aggregate97-call/token witness is not an
individual layer-ID trace. Own-arm warm/timed replay stays bitwise.

All original native gates pass, with no numerical-threshold revision:

- All32 full-vocabulary rows have matching argmax. Minimum cosine0.999999999240,
  maximum relative RMS3.909643e-5, maximum absolute delta7.624626e-4.
- Hyper maximum RMS2.342685065e-5, maximum absolute1.302957535e-4.
- All121 persistent tensors pass. F32 maximum RMS1.491637686e-5, maximum absolute
  2.956390381e-4; newly appended F16 rows maximum RMS1.296085797e-4, maximum
  absolute0.00390625. New rows are checked separately to avoid prefix dilution.
- Old prefixes remain immutable and unused suffixes agree. All values finite.

Raw baseline/candidate matrices, each31,784,960 bytes, were saved before candidate
census/numerical gates in `target/profiles/qwen4exp-native-hc-up-23497/`.
The reused comparison helper labels log rows `native_split`; in this packet the
changed arm is HC, while split-QSA remains enabled in both.

## Incremental Parent Timing

One frozen warm four-forward ABBA followed by one measured four-forward ABBA,
from the same checkpoint. Restore/readback/assertion work is outside clocks.
Wall values sum executor timings, not external request latency. API validation
is enabled symmetrically; no shader-validation instrumentation claim.

| Four-forward metric | A1 ms | B1 ms | B2 ms | A2 ms | Time saved | A spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 193.693500 | 183.868625 | 184.009042 | 194.670708 | 5.275085% | 0.503243% |
| Executor wall | 215.273000 | 204.555957 | 205.351875 | 216.268710 | 5.013160% | 0.461466% |

Both metrics pass frozen mean AND pairwise floors (GPU5%, wall3%), with controls
within5%. GPU pairs save5.0724% and5.4768%; the first narrowly clears the floor.
Mean GPU saving is about2.561ms per forward. Do not combine the38.2602% component
percentage with this native percentage, or convert it into generic throughput.

The prefix observes25,163.46ms executor wall /4,169.25ms GPU across three commands.
That mixed first-use observation is not a balanced prefill speedup or OS-cold claim.
Earlier aborted native runs remain failures before HC qualification; this new
packet supplies the native authority.

## Recharted Leverage

1. Deliver the demonstrated HC benefit through a narrow default-off option,
   initially qualified together with split-QSA. Preserve the unchanged shader,
   Q8/four-branch/hidden2560/rank320 scope, incumbent fallback and existing scratch.
   Use actual product bindings for the bounded32-token and four-forward packet.
   Independently selectable HC without split needs its own composition check.
2. Make complete-MoE observation the next research step before another expert
   topology. Include routing, expert work and combination on representative native
   decode state. Do not make this a prerequisite for delivering qualified HC.
3. Keep HC-body, attention-body and RMS sweeps parked. Context-growing index
   scoring remains separate from the2179-token budget.

Production CLI `cargo check` passes. Read-only reviewer
`01a0a13a-f2f8-7673-ace3-4fbfd25a3aef` confirms numerical/timing gates and supports
default-off delivery next, not default promotion or broad model-quality authority.
No server was stopped/restarted in this follow-up; the previously stopped server
remains stopped. No remote push.

## Raw Evidence

Optimization worktree, `target/profiles/`:

- `2026-09-15-qwen4exp-native-hc-validation-trace-02.log`: IQ4 down localization.
- `2026-09-15-qwen4exp-moe-down-index-{narrow-511,narrow-512,wide-512,populated}.log`;
  `2026-09-15-qwen4exp-moe-down-index-product-regression.log`.
- `2026-09-15-qwen4exp-packed-expert-inventory.log`: CPU metadata inventory.
- `2026-09-15-qwen4exp-{iq4-xs,q8-down}-index-original-512.log`, plus the corrected
  `2026-09-15-qwen4exp-q8-down-index-original-512-corrected.log`.
- `2026-09-15-qwen4exp-remaining-index-{paired,product-regression}.log`.
- `2026-09-15-qwen4exp-native-hc-up-03.log`: still-blocked prefix;
  `2026-09-15-qwen4exp-native-hc-up-04.log`: current numerical/performance PASS.

Original narrow output SHA256s retained in the product regression tests:

| Capture | SHA256 |
|---|---|
| IQ4_NL down,10547 | `7c36d7a8a5de6bb555b9db8d6111fc59b06e81c496c6577ebf1aafe16b253062` |
| IQ4_XS gate/up,21508 | `49c3d0b8ebca5a7d1fa78535835446d6b199bd40bfc21141030804d983e3d933` |
| Q8 down,21508 | `f9a505f4ce5e8216b078d570f593fbe46e5c67022bb8b7851afc52daecf0b029` |

These are local M4 captures, not promised cross-device/compiler bitwise oracles.

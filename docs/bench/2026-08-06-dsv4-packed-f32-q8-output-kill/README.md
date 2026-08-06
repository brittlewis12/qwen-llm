# DeepSeek V4 Packed Q8 F32 Attention Output KILL

Status: decisive current-asset `KILL`. The fixed F32 matrix arithmetic preserves
the model-free speed ceiling from `c74d791`, but it still changes the deployed
Q8 reduction lineage. Repeating that perturbation through 43 layers crosses
consumed MoE decisions and misses the frozen deep hidden-state gates. The sole
authorized packet also misses aggregate wall saving and candidate stationarity.

The live experiment source is removed. Ordinary packed execution remains on the
exact deployed output path. The committed model-free F32 kernel, active reduced
tests, and sealed synthetic profiler remain as local arithmetic evidence only.

## Question

The prior F16-staged Q8 matrix schedule cut the synthetic N=128 output stage by
75.525%, but rounded reconstructed Q8 weights and F32 activations through F16.
Its canonical 128+12 current-asset packet changed packed and continuation routes
and reached 0.070735848 relative RMS in the position-140 normalized hidden
vector.

Commit `c74d791` removed both half operand conversions. On the same synthetic
production shape, A+B relative RMS fell from 0.001230151 to 0.000005573, a
220.7x repair, while GPU median saving remained 72.2179%. This packet asks
whether that materially different arithmetic is accurate and fast enough after
full-model amplification.

## Frozen Candidate

The candidate uses the single model-free-qualified F32 `R2C4K64` Q8 matrix
geometry:

- one 32-thread SIMD group;
- 16 output rows by 32 token columns;
- K-step 64 and eight F32 matrix accumulators;
- Q8 values reconstructed as F32 scale-times-quant;
- activations loaded directly as F32; and
- no half matrix operands or `fast::` operations.

The test-only integrated policy selects F32 matrix output A and B only when the
chunk begins at position zero and contains exactly 128 tokens. A following
12-token chunk and the restored singleton continuation use the exact deployed
path. Transactional counters require 43/43 candidate A/B invocations after the
first chunk and no later change; every control must remain 0/0.

The canonical natural-language prompt contributes exactly 140 tokens. Its
token-ID SHA-256 is asserted before residency planning or loading:

`8dc3a3091bc13e6f8e982e25e251e3995565d26b89e70a1f7cc8da36060de2f7`

One untimed detailed control/candidate pair captures positions 128 and 140,
snapshot metadata, exact packed route schedules, learned rank-6/rank-7 margins,
and one route-only singleton transcript after canonical restore. Timed execution
then uses C/A/C/A/C. Observation, snapshot, and transcript work is outside the
timed chunk intervals.

Frozen gates are unchanged from the prior campaign:

- exact argmax at positions 128, 140, and restored continuation;
- cosine at least 0.999 and relative RMS at most 0.05 for full logits and final
  normalized hidden vectors at all three checkpoints;
- exact packed expert IDs, stable expert-major schedules, and continuation IDs;
- zero sparse CSA selectors at this shallow prefix;
- at least 20% first-chunk pre-expert GPU saving;
- at least 15% aggregate pre-expert GPU and wall saving;
- no more than 5% exact-tail GPU regression; and
- no more than 5% within-arm GPU or wall drift.

CX statically reviewed the compiled harness and authorized exactly one run after
the canonical token digest became a pre-load assertion. No code, asset, or
configuration changed between that GO and execution.

## Result

Invocation ownership and snapshot metadata pass. Controls and candidates repeat
their observations and continuation route transcripts bit-for-bit. One detailed
pair captures packed decisions for cross-arm audit. All three argmaxes remain
305 / 12,122 / 20,332.

### Quality

| Checkpoint | Vector | Cosine | Relative RMS | Gate |
|---|---|---:|---:|---|
| 128 | logits | 0.999630906 | 0.027230466 | pass |
| 128 | normalized hidden | 0.999297221 | 0.037484206 | pass |
| 140 after exact tail | logits | 0.998981632 | 0.045150382 | fail cosine |
| 140 after exact tail | normalized hidden | 0.997783384 | 0.066547844 | fail both |
| restored continuation | logits | 0.999105050 | 0.042305721 | pass |
| restored continuation | normalized hidden | 0.998619943 | 0.052519175 | fail both |

Removing half operand conversion improves integrated relative RMS by only about
6-16% versus the prior half/half packet, despite the 220.7x local synthetic
repair. Fixing operand precision was therefore insufficient to control
full-model error; the packet does not isolate a unique dominant amplification
mechanism.

### Decisions

Both arms execute zero sparse CSA selectors. Packed expert IDs and stable
expert-major schedules differ, and the restored singleton continuation changes
expert IDs as well. Maximum packed route-weight delta is 0.352247149; maximum
continuation delta is 0.129800946. Minimum learned rank-6/rank-7 margin falls
from 0.000021935 in control to 0.000002861 in candidate, with maximum margin
delta 0.304895401.

The matrix changes the deployed block-scaled scalar reduction order even with
F32 operands. Small continuous differences traverse residual and
hyper-connection state until near-tie learned routing becomes discrete; changed
experts then amplify the perturbation. The exact tail cannot repair causal state
already committed, and snapshot restore correctly preserves it.

### Performance

| Metric | Controls (ms) | Candidates (ms) | Decision value |
|---|---|---|---:|
| First-chunk pre-expert GPU | 1090.833 / 1089.983 / 1104.951 | 802.134 / 798.956 | 26.554% saving |
| Aggregate pre-expert GPU | 1314.906 / 1318.258 / 1328.748 | 1026.869 / 1022.713 | 22.064% saving |
| First-chunk wall | 3719.538 / 3750.794 / 3744.620 | 2979.582 / 3322.381 | support only |
| Aggregate wall | 4460.497 / 4498.744 / 4489.260 | 3722.146 / 4063.653 | 12.725% saving |
| Exact-tail pre-expert GPU | 224.073 / 228.276 / 223.797 | 224.735 / 223.756 | 0.201% regression |

GPU drift passes at 1.047% control and 0.406% candidate. Control wall drift
passes at 0.854%; candidate wall drift is 8.773%, above the 5% limit. Aggregate
wall saving is 12.725%, below 15%. Performance therefore fails independently of
quality and decision correctness.

## Decision

Close packed Q8 attention-output replacement on the current asset and Apple M4
Max. Exact mapped T1/T4/T8 schedules have no economic win; F16-staged matrix
arithmetic fails quality and routing; F32-operands matrix arithmetic now fails
the same full-model contract. Do not rerun, retile, lower the chunk threshold,
relax the gates, add an opt-in, or attempt another same-family rescue without a
materially different model, device, or arithmetic premise.

Remove the ignored current-asset packet, integrated first-chunk policy,
invocation counters, packed decision observers, route-only singleton mode, and
their plumbing. Preserve the exact deployed output path and the committed
model-free F32 kernel/tests as bounded negative evidence.

The next packed-prefill branch is an exact joint chronological-publication and
attention-body rewrite. It may share memory boundaries that were individually
below threshold, but it requires a new model-free ceiling rather than adding old
sampled estimates. A direct production-shape grouped MXFP4-down ceiling ranks
second; further HCA tiling remains deferred until attribution makes it leading.

## Provenance

- Base revision: `c74d791ddb30ba436eceec0ca122e5a388bd8563`.
- Device: Apple M4 Max.
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`,
  104,207,848,032 bytes across four census-pinned shards.
- Canonical packet elapsed time: 67.80 seconds.
- Exact experiment source diff SHA-256:
  `4af60e6a1f88586b46ff870283def64b113254c9227a391be2498f6d13dc654e`.
- Raw current-asset log SHA-256:
  `2d1c8039ad11024ee83f1e25ad99f74214d099ab6d18124666849e0d1bac5e04`.
- Executed release test binary SHA-256:
  `0e467f71267dd7b0bb070b26fdcf953ffd7fc3b45a8882541a478024002a2426`.
- Control first/tail/continuation logit SHA-256:
  `71d31fe3...d214ea7`, `25bffc01...64f1415`, and
  `3e494d8e...d611390`.
- Candidate first/tail/continuation logit SHA-256:
  `a9282ee9...edbff47`, `7c56d330...48b1e4a`, and
  `ac6b2aa5...68d551`.

The standard unified source diff reconstructs the complete diagnostics-only
experiment from the exact base revision. It was captured before cleanup. The
raw log and executed binary were hashed before any rebuild.

Pre-execution checks:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_f32_q8_first_chunk_packet \
  --no-run

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  q8_f32_mma_r2c4k64_

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  route_only_capture_is_complete_without_sparse_selection

cargo check --release -p qwen-llm --lib
```

Sole current-asset command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_f32_q8_first_chunk_packet \
  -- --ignored --exact --nocapture --test-threads=1
```

CX review and adjudication: `019fd56e-3f17-7491-aaf2-07dd5bee62c6`.

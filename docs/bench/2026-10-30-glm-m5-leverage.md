# GLM-5.3-Flash: Focused M5 Leverage Map

## Status

Source/evidence review against `adad4461`, not a new benchmark or a claim of
M5 saturation. GLM and V4.1 both have substantial default M5 optimization;
neither is a completed reference implementation for the other. Compare gains
against each family's own baseline, not cross-model token rates.

The recent usage artifact is GLM UD-Q4_K_XL. The dense/down landing packets
below use stock UD-IQ3_XXS; older first-light measurements use Maya-M IQ2_S.
Their quantization inventories and selected paths are not interchangeable.

Subsequent [actual-M5 native transfer measurements](2026-10-30-glm-m5-transfers/README.md)
on six-shard UD-Q4_K_XL are now a **performance-only checkpoint**, not a
qualification or source-integration verdict. Independent wide-only and
Q5-down-only blocks reduce median 2K/4K native wall by 3.3526%/3.1844% and
8.3784%/7.5065%; N512 does not improve. The packet retains all 48 samples,
late-run drift and the A0 CPU-observer caveat. Its SDK 26.2 build is distinct
from V4.1 SDK 27 serving results. Strict numerical qualification remains
pending; no flag/default promotion follows. The historical map below is
unchanged, not a claim that these native timings measure serving TTFT.

## Measured Anchors

- [Dense tensor prefill](2026-10-22-glm-tensor-prefill/README.md): M5 Ultra,
  Fast/Half, Q6_K/Q8_0, at least 1024 committed rows. Ordinary 2K/4K prefill
  wall falls 15.1%/14.3%; the saved 11K request's prefill falls 12.7%.
- [Tensor down](2026-10-23-glm-tensor-down/README.md): IQ3_S down at exactly
  1024/2048 rows. Against dense-only, ordinary 2K/4K wall falls 9.1%/8.7%;
  saved-request prefill falls 6.1%. These are sequential changes with distinct
  controls, not additive percentages.
- The latter packet's M5 stage profile is dense-only, before tensor-down:
  routed experts 56.9%, MLA projection 13.4%, KDA input/output projections
  9.4%/3.1%, recurrence 6.2%. Profiling adds 6.7% wall overhead; these are
  diagnostic span shares, not current recoverable wall fractions.
- Earlier M4 bandwidth and stage maps are historical context, not evidence
  that current M5 decode or prefill has reached its ceiling.

## Candidate Leverage

### Shared Dense Operands

`glm5_next_metal/tensor_prefill.rs` explicitly selects the qualified narrow
F32-input tensor route. Shared half-activation staging and the newer shared
Q8_0 tile are candidates for a bounded transfer, not missing tensor support.
Keep eligibility by actual dtype, geometry, arithmetic and committed rows;
price any staging and include its preparation in the complete operator cost.
Do not change strict modes or import a V4.1 speedup estimate.

### Residual Routed Pipeline

The down landing leaves 84 fused gate/up dispatches per full chunk and three
IQ4_XS down calls unchanged on its artifact. Refresh the parent expert budget
before splitting it into route preparation, bucketing, useful/padded matrix
work and scatter. Existing parallel bucketing is a possible transfer only if
its inclusive cost beats the current composition. Do not restart the negative
M4 wider-down or lower-state gate/up sweeps merely because hardware changed.

### Q8 MLA Ragged Tails

The Q8 MLA absorption/expansion path batches complete 128-row groups, then
dispatches grouped GEMV per remaining token. A token-axis grouped GEMV can
retain each dot's arithmetic while consolidating remainder dispatches. The
Q6 path already handles scalar remainder rows differently, so this is
dtype-specific. The earlier tail-MMA experiment remains off: its useful M4
timings do not resolve the recorded code-output differences.

## Bounded Next Measurement

Use the existing harness and profiler; do not create a new benchmark framework.
Pin the actual artifact, precision, source, admitted workspace and execution
paths before comparing anything.

Start with cold N512/N2048 and one short/ragged append at a representative
existing depth. For cached latency, distinguish immediate and idle calls and
live reuse from snapshot restore. Record ordinary wall/TTFT separately from
unsplit GPU/driver intervals, capture/restore and diagnostic stage spans.
Run census/profiling outside plain timing; do not add overlapping intervals.

Choose one leaf only after that ranking, then check complete-operator and
ordinary-request conversion. Preserve numerical policies and use existing
continuation/tool checks when arithmetic changes. A leaf win that does not
convert is not promoted. No model switches, measurements or GPU reservation
were performed for this map.

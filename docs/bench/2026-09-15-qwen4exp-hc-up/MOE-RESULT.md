# Complete MoE Observation: Timing Invalid

Observation01, API validation and production lease/real wired gate,5.10s:
`target/profiles/2026-09-15-qwen4exp-moe-observe-01.log`.
One2179 prefix, QSA on/HC off, ordinary and observed forward at2179. Scoped
observer visits all48 MoE calls, identifies layers2/4/5 by router identity and
ordinal, copies native input/output/routes before reuse. Full logits, hyper and
all121 persistent tensors are bitwise; raw full logits saved under
`target/profiles/qwen4exp-moe-observe-905/`.

Layer2 IQ4_XS/Q8 native and warm replay census match eight dispatches. Warm,
unsampled-before and sampled outputs/routes pass finite/bitwise checks. Then the
192-counter packet fails global cross-encoder monotonicity. Length/sentinel gates
passed, but individual positive-span and sampled-census gates were not reached.
Layers4/5 replay and unsampled-after were not reached. No timing values or
complete-stage attribution are qualified. Effective IQ3/IQ4-down fast routes on.

Persistence was too late: raw counters and partial timings were not saved before
the assertion. Fix packet-level persistence before gates; retain the original
failure, do not relabel it a successful run. The original MoE gate remains and
the full packet has not been rerun.

Apple's serial dispatch guarantee is within a compute pass, not a guarantee of
globally disjoint sampled intervals across encoders. Overlap is plausible, not
proven for the failed packet. Separate frozen model-free interval diagnostic:
96 dependent copy encoders,192 valid samples, all individual spans positive,
zero adjacent overlaps and zero reversed starts; copies bitwise PASS0.04s.
`target/profiles/stage-interval-diagnostic-3783.json` saves all raw counters.
This control does not reproduce or explain the native packet's anomaly.

Next instrumentation investigation can reuse saved layer2 input/output/routes
without another full-prefix qualification. Classify individual invalid spans
versus cross-encoder overlap and persist evidence first. Do not assign exclusive
costs by adding overlapping spans or normalize away encoder overhead. No kernel
candidate or new performance ranking is justified yet. HC promotion stays HOLD.

## Saved Layer2 Diagnostic: Reordered Disjoint Spans

One separate frozen diagnostic loads SHA-pinned saved layer2 input/output/routes,
loads the existing weights, but executes no prefix/native forward. The16 sampled
complete chains PASS0.20s: all192 samples valid, every individual span positive,
input unchanged and output/routes bitwise. All16 eight-dispatch sequences match
on offline review. Raw counters and census persisted before gates in
`target/profiles/saved-moe-interval-diagnostic-4380.json`.

Three start-order reversals at pair indices2/8/86 map to routed-down -> shared
gate/up. Important classification correction: the artifact's original
`adjacent_overlap_indices` test only meant next_start < previous_end. Inspection
shows shared gate/up actually ENDS before routed-down STARTS in all three cases:
reordered disjoint intervals, not overlap. Preserve the original artifact and
correct the classifier using max(starts) < min(ends), with a CPU unit test.

This establishes that encoded order is not a valid global timestamp ordering
assumption for these independent branches; it does not recover missing counters
or qualify observation01. Next use saved layer2 under a versioned fixed observation
protocol: valid individual spans, exact census, inclusive durations and sorted
interval union/envelope/overlap. Keep any command-time scaling labeled normalization,
not calibration, and avoid exclusive-cost/unattributed-time claims. Layers4/5
replay and a three-class budget remain pending. No new HC timing or kernel sweep.

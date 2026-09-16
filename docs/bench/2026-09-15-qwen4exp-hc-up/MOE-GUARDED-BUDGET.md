# Remaining MoE Budget With Guarded Routing

Frozen before execution. Guarded top-k product/CLI qualification is already PASS;
this new baseline observation is not a selector timing retry. QSA ON, guarded
top-k ON, HC OFF. Old serial-router costs no longer rank the remaining work.

One2179 prefix, ordinary and observed one-token forwards from identical checkpoint.
Capture layers2(IQ4_XS/Q8),4(IQ3_XXS/Q8),5(IQ3_XXS/IQ4_NL), all native inputs/output/
selected IDs/weights. Ordinary/observed logits/hyper/terminal121states bitwise.
All48 native singleton calls use guarded product kernel. Persist all three captures
before any per-layer replay/timing, avoiding the original packet's evidence loss.

Private replay workspace explicitly binds guarded top-k; merely enabling the loaded
model option would not configure these independent workspaces. First qualify all
three complete-MoE outputs/routes bitwise, finite inputs, unique in-range IDs and
exact native/isolated census (8/8/9 dispatches). Then per layer three further warm
chains,16 unsampled/16 six-stage sampled/16 unsampled chains. Four total warm chains
including qualification. Fixed schedule, no retry, drift>5% INCONCLUSIVE.

Use existing interval-aware V2 accounting: positive individual spans, sorted union/
envelope, overlap multiplicity and concurrent wall ticks. Inclusive stage fractions
and command-span normalized times are not exclusive costs or clock calibration.
Save raw counters/status/census before gates; all measured packets match native
census and bitwise output/routes, input remains unchanged. No kernel candidate.

Production lease/real wired-memory check before Metal, API validation on. This is
warm isolated replay of three representative native inputs, not an all-layer
average or full-model attribution. Rank the next source mechanism only where the
observed parent cost and dtype coverage support it. HC performance HOLD unchanged.

## Result And Rechart

Observation01 completes in5.10s with native logits/hyper/terminal121states bitwise.
All three private replays preserve output/routes and input bytes; exact8/8/9
censuses match native in every recorded packet. All captures, counters and results
are retained in `target/profiles/qwen4exp-moe-observe-guardedtrue-98931/`, log
`target/profiles/2026-09-16-qwen4exp-guarded-moe-budget-01.log`.

| Layer /dtype | Unsampled before ms | After ms | Drift | Decision |
|---|---:|---:|---:|---|
| 2 IQ4_XS/Q8 | 0.743958 | 0.280214 | 90.5600% | INCONCLUSIVE |
| 4 IQ3_XXS/Q8 | 0.165555 | 0.167805 | 1.3499% | Bounded |
| 5 IQ3_XXS/IQ4_NL | 0.145440 | 0.146477 | 0.7101% | Bounded |

Times are per warm isolated complete MoE chain, not native layer attribution.
No retry of layer2. Qualification and the three later warm chains are separated
by other layers' qualification work; they are not four consecutive warm chains.

Normalized inclusive sampled ms for the stable rows:

| Layer | Router group | Routed gate/up | Down/sum | Shared gate/up | Shared down | Accumulate |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 0.049089 | 0.045549 | 0.050396 | 0.010086 | 0.011664 | 0.002552 |
| 5 | 0.049314 | 0.043853 | 0.035645 | 0.008302 | 0.011653 | 0.002594 |

Sampled command envelopes0.181273/0.168182ms differ from unsampled totals;
stable controls do not prove instrumentation is unperturbed. The router group
still includes projection, selection and shared sigmoid, not selector-only cost.
The large sampled IQ4_XS gate/up belongs to one layer with unstable controls;
do not prioritize a47-layer IQ3 rewrite from that outlier.

Next highest-leverage step is one small current-baseline whole-forward ledger,
using existing checkpoint/layer-profile machinery. Distinguish complete GDN-
containing blocks, QSA-containing blocks, bootstrap and tail against ordinary
execution. Those parents still include HC/MoE/projections, not recurrence-only
or attention-only work. Inspect the largest remaining parent's source before
choosing another expert kernel. Common IQ3 gate/up and IQ4_NL down remain
conditional candidates, not demonstrated wins; long-context QSA indexing is a
separate regime. Keep HC HOLD and further selector/body sweeps parked.

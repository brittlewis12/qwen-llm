# GLM small-append tensor expert limiter packet

Status: limiter packet complete; K128 candidate failed the 1.25x routed-stage
gate and was removed. Production behavior is unchanged.

Branch: `perf/glm-tensor-experts`, based on main `f691af6e1` (main was inspected
before creating the worktree). Stock GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra,
81,920-token code prefix, 98,304 capacity, 2,048-row prefix chunks.

## Measurement method and two corrections

`glm_small_append_price ... occupancy 81920`, built with `--profile iter
--features perf-ablations`, retains per-layer bucket counts and every expert
dispatch's layer, grid, threads and timestamp. The original scalar bandwidth
fields are historical; this packet recomputes attribution from the dispatches.
A gate projection's pipeline is named `moe_down_iq2_s_*`, so counting only the
`moe_up_*` pass as gate/up doubles its apparent bandwidth and adds the gate's
time to down. The new report classifies both gate and up together.

There is also a route mismatch with the lane's premise. At 161/256/512 rows,
39 blocks pass `Iq3STensorOrSmallCounts` into the explicit adapter, which only
accepts `Iq3STensor`. Those blocks run the F32-storage tensor gate/up; the three
IQ4_XS-down exception blocks run the explicit Half-storage composition. At 32
rows all blocks use the explicit composition. Both tensor variants stage Half
operands. The observed main behavior is retained as the performance reference.
This conflict was reported on `SRV-2K0Y`; it is not silently relabeled as the
requested all-explicit route.

| Rows | Routed stage ms | Gate/up ms | Down ms | Gate/up GB/s | Down GB/s | Stage GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 116.20 | 70.42 | 44.55 | 367.4 | 392.8 | 373.2 |
| 161 | 224.25 | 138.52 | 82.57 | 361.4 | 409.9 | 374.2 |
| 256 | 261.06 | 158.38 | 96.51 | 345.2 | 383.3 | 351.1 |
| 512 | 343.69 | 206.31 | 123.99 | 287.5 | 323.6 | 289.4 |

These are one stage-profiled append per width and a separate dispatch census,
not paired performance-gate medians. The kernel census serializes dispatches;
its attribution is diagnostic rather than an ordinary wall-time claim. Rates
use decimal GB and count each active expert once per layer, including the
IQ3_S gate/up and IQ4_XS down exceptions.

## Grid, buckets and threadgroup memory

| Rows | Active experts/layer | Bucket p50 / p90 / max | Gate/up groups launched/expert/pass | Active fraction of launched gate panels | Active N32 capacity / useful slots |
| ---: | ---: | --- | ---: | ---: | ---: |
| 32 | 113.6 | 1 / 4 / 27 | 32 | 39.45% | 14.20x |
| 161 | 219.8 | 4 / 12 / 126 | 192 | 12.89% | 5.53x |
| 256 | 240.1 | 5 / 18 / 202 | 256 | 10.83% | 3.90x |
| 512 | 260.5 | 10 / 34 / 402 | 512 | 6.57% | 2.37x |

The full histograms and each layer's counts remain in the raw census. Pooled
expert-layer bins (12,096 bins per width) make the small-bucket mass explicit:

| Rows | Empty | 1 row | 2-4 rows | 5-8 rows | 9-16 rows | 17-32 rows | 33+ rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 60.55% | 19.80% | 15.86% | 2.93% | 0.77% | 0.08% | 0.00% |
| 161 | 23.69% | 14.85% | 29.23% | 17.10% | 10.81% | 3.45% | 0.88% |
| 512 | 9.54% | 6.27% | 17.20% | 17.58% | 22.59% | 16.89% | 9.93% |

Gate/up has two dispatches/layer. Each uses 128 threads, grid
`(ceil(rows/32), 32, 288)`, 8 KiB dynamic threadgroup memory. The 32-row down
uses `(1,64,288)`, 128 threads and 8 KiB. At 161/256/512 rows, each IQ3_S-down
block has three disjoint count bands: N16 `(1,64,288)` with 5,120 B, N32
`(1,64,288)` with 8,192 B, and N64 `(ceil(rows/64),64,288)` with 16,384 B.
Each IQ4_XS exception block has one down dispatch, so the average down count is
120/42 = 2.857 dispatches/layer. Routing and weighted reduction add two
dispatches/layer, plus the explicit composition's Half conversion where used.
There is no dispatch per individual expert.

The device exposes a 32 KiB threadgroup limit and only the `timestamp` counter
set (`GPUTimestamp`). Static threadgroup memory and maximum threads are recorded
per pipeline: the measured PSOs report zero static bytes, SIMD width 32 and
maximum 1,024 threads/group. The allocation-only ceilings are 4 / 6 / 2 resident groups for
N32 / N16 / N64; these are not measured occupancy and exclude registers,
thread-count limits and scheduler constraints. Live occupancy, register spills,
ALU utilization, cache traffic and GPU clock counters are unavailable through
the exposed counter sets. There are already thousands of active groups/layer;
a lack of total parallel work is not established.

## Counterfactuals

The standalone C++/Metal diagnostic is outside the production library. Both
arms use the product's `metal4.0@26.0 -O3 -ffast-math` flags. It replays each
captured 80K per-layer histogram against full 288-expert synthetic native banks,
with deterministic synthetic activations and permuted expert placement. It
retains gate N32 and down N16/N32/N64 policies, including the full rectangular
launch grid. It is a distribution-matched synthetic experiment, not a replay
of actual model weights or activations. Each arm has three rotated repetitions,
with four dispatch sequences per timing sample. Tiny M5 smoke checks and the
full-shape controls compare to the maintained production kernel at the existing
`0.002 * reference_max + 0.00001` tensor witness bound, also recording bit
mismatches. Predecoded F16 weights use the unmodified native dequantizer.

- Full (`h0`): the generalized staging control, including MMA.
- No MMA (`h1`): volatile shared Half writes retain every dequantized value and
  both K-loop barriers. A final observable shared read prevents dead-code
  elimination. This avoids the first attempt's checksum reduction in the hot
  loop. Volatile staging is an additional confound, so this is not an exact
  subtraction of MMA time.
- No dequant (`h2`): stream the canonical predecoded full F16 bank, retaining
  staging and MMA. This increases weight bytes by 6.24x (IQ2_S) or 4.65x
  (IQ3_S), so it is a counterfactual, not a free-dequant lower bound.
- Constant weights (`h3`): retain staging and MMA without quant reads/decode.
- Resident quant blocks (`h4`): retain native decoder arithmetic but reuse the
  first 256-element quant block as K advances. This removes most external
  weight traffic without removing decoding or scale/codebook instructions.
- Empty: the unmodified production pipeline with zero bucket counts and the
  unchanged full launch grid. This prices the empty return/dispatch tail.

The generalized full control costs 11-20% more than the unmodified production
kernel. Counterfactual interpretation uses `h0` as its matched reference;
production-relative numbers are also retained. The first no-MMA checksum
control and an interrupted gate-policy setup are explicitly excluded from
limiter conclusions. Counterfactual differences overlap and must not be added
as percentages of a single critical path.

Each projection below sums the 42 per-layer medians on that synthetic bank
type; it is not the mixed-dtype model total. Ratios use the matched full
control, except the empty-grid column, which uses production.

| Rows | Projection | Production ms | Full control ms | No MMA / full | F16 no dequant / full | Constant / full | Resident quant / full | Empty / production |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | iq2_s | 34.77 | 40.53 | 0.719 | 2.719 | 0.634 | 0.937 | 0.011 |
| 32 | iq3_s | 39.78 | 44.57 | 0.729 | 2.297 | 0.564 | 0.884 | 0.017 |
| 161 | iq2_s | 65.05 | 77.54 | 0.715 | 2.755 | 0.625 | 0.917 | 0.029 |
| 161 | iq3_s | 74.23 | 83.63 | 0.780 | 2.425 | 0.442 | 0.837 | 0.044 |
| 512 | iq2_s | 93.32 | 112.32 | 0.710 | 2.512 | 0.611 | 0.894 | 0.053 |
| 512 | iq3_s | 113.19 | 126.23 | 0.746 | 2.008 | 0.503 | 0.860 | 0.056 |

All 672 full-shape full-control/predecoded comparisons were bitwise equal to
the unmodified kernel. The 256-row cohort is retained in raw data and the JSON
report; it follows the same pattern.

**Limiter decision:** work inside active tiles dominates. Empty launch/return
cost is only 1-6%, and the resident-quant experiment removes external weight
streaming for a 6-16% reduction against its matched control. Removing decode
and weight reads with constants saves 37-56%, while the no-MMA twin saves
22-29%. This implicates native dequant/scale/codebook instructions and shared
tile staging/barriers, with N32/N64 padding spending additional MMA capacity.
It does not identify external scale/metadata byte bandwidth as the bottleneck,
nor does it establish grid underfill or per-expert launch latency. Without
hardware utilization counters, decoder ALU cannot be separated cleanly from
cache-local scale/codebook load instruction throughput. Threadgroup memory is
a capacity constraint, not a measured occupancy diagnosis.

## Candidate chosen from those measurements

The experimental K128-staging kernel decodes each 128-element slab into shared
Half storage, then performs four ordered K32 tensor MMAs before replacing the
slab. It uses M32 output tiles, 256 threads/eight SIMD groups, N8 for buckets
through eight rows and N32 above. Two disjoint count-band dispatches loop over
each expert's actual panels. Dynamic threadgroup memory is 10 KiB / 16 KiB;
there is no persistent buffer. This attacks K-loop barrier frequency, wider
parallel dequant and small-bucket padding together. It retains canonical
Half values and K32 accumulation steps; changed tensor geometry still requires
numerical qualification if the speed gate succeeds.

The standalone synthetic screen is bitwise equal on 336 full-shape candidate
comparisons. Its estimated two-gate-projection-plus-down speedup is
1.04x / 0.94x / 0.89x / 0.82x at 32 / 161 / 256 / 512 rows. These sums are
screening evidence, not the required full-model routed-stage gate. The full-model
paired stage measurements below determine the final decision.

## Full-model gate and decision

Source `45e3ae975`, same `iter` binary for both arms, stock model and the same
81,920-token snapshot. At each width, warm both arms, then three interleaved
blocks: ABBA / BAAB / ABBA (six samples/arm). The table reports medians. The
90% intervals resample the three paired block speed ratios, 20,000 resamples;
with only three blocks, they describe this packet rather than a population.
The interval statistic is a paired-block geometric mean; the speedup column is
the ratio of arm medians, so they need not coincide exactly.

| Rows | Main expert stage ms | K128 stage ms | Speedup | Paired-block speed 90% CI |
| ---: | ---: | ---: | ---: | --- |
| 32 | 115.88 | 110.72 | 1.047x | [1.046, 1.047] |
| 161 | 225.08 | 236.04 | 0.954x | [0.946, 0.953] |
| 256 | 260.67 | 286.88 | 0.909x | [0.908, 0.910] |
| 512 | 343.05 | 406.90 | 0.843x | [0.843, 0.843] |

The 161-row dispatch census records 246 `glm_k128_*` calls in the candidate
arm: two count bands for each of 84 gate/up projections and 39 IQ3_S down
projections. IQ4_XS down retains its incumbent kernels. The baseline has no
candidate dispatches. Thermal pressure was 1 for the complete model A/B;
background `ghost` work remained resident. The measured build took 35.77 s.

**Gate: fail at all three required widths.** The candidate is 4.9% / 10.1% /
18.6% slower at 161 / 256 / 512, respectively, instead of at least 1.25x
faster. The 4.7% win at 32 does not qualify the real median-width band. Wider
K staging and N8 panels did not reduce decoder instruction count, and their
extra shared storage, changed thread geometry and per-expert panel loop did
not convert the active-tile headroom into a speedup. The experiment cannot
separate those effects without more counters; it is not evidence that every
small-panel design must lose.

The kernel, experimental selector, full-model A/B mode and standalone candidate
arm were removed. The diagnostic holds, raw grid capture and corrected
bandwidth attribution remain. No new default or environment rollback knob is
shipped. The explicit-adapter policy mismatch is recorded and left unchanged;
changing that route alone is not a qualified 1.25x kernel result.

The performance gate stops promotion: no ordinary wall campaign, paired NLL
non-inferiority campaign, release build or three-agent serve A/B was run. The
336 bitwise synthetic operator comparisons are kernel witnesses, not a claim
about the paired model-quality protocol. To reproduce the removed candidate,
use historical commit `45e3ae975` and mode `tensor-ab 81920`; the final branch
tip retains only the measurement tools and this negative result.

## Reproduction and constraints

Raw records are at `metagross:~/qwen-runs/2026-11-12-glm-tensor-experts/`.
`census-current.jsonl` is the fresh main census; `route-source.jsonl` preserves
the earlier distribution source; `holds.jsonl` is the corrected no-MMA cohort;
`empty.jsonl` prices empty dispatches. `*.observer.jsonl` records thermal
pressure, load and exact child PIDs; `*.processes` captures background load.
`ghost` processes were left running. There is no quiet-host claim. The missing
fixture-path attempt exited before any model load. Bundles move commits into a
separate M5 worktree; every M5 build/GPU packet queues on the canonical lease,
with declared holds of 8-12 minutes and bounded model/diagnostic execution.

```sh
# Every M5 command below is run inside the canonical lease with nice -n 10.
cargo build --profile iter --features perf-ablations --example glm_small_append_price
GLM_ROUTE_WIDTHS=32,161,256,512 GLM53_GGUF="$MODEL" \
  "$TARGET/iter/examples/glm_small_append_price" "$CORPUS" occupancy 81920
python3 scripts/bench/glm-tensor-experts/build.py "$OUT/controls"
"$OUT/controls/probe" "$OUT/controls/probe.metallib" "$OUT/census-current.jsonl" holds
"$OUT/controls/probe" "$OUT/controls/probe.metallib" "$OUT/census-current.jsonl" empty
python3 scripts/bench/glm-tensor-experts/report.py "$OUT"
```

The M4 compiled the standalone Metal controls. Its OS cannot load Metal 4
libraries, so actual tensor execution is M5-only. Existing synthetic native
IQ2_S gate/up and IQ3_S down GPU tests passed on M4 in the test profile. No new
ignored test was added. The ENV reference is regenerated; contract validation
and cleanup are recorded below.

## Validation and cleanup

M4 commands (optimized test profile, `nice -n 10`):

```sh
cargo test -p qwen-llm --lib generic_grouped_moe_gate_up_iq2_s_matches_cpu -- --nocapture
cargo test -p qwen-llm --lib generic_grouped_moe_down_iq3_s_matches_cpu -- --nocapture
```

Each ran one GPU test and passed; their logs are in the raw packet. The final
example is checked in the `iter` profile. Canonical validation is
`nice -n 10 scripts/check-contracts.sh`; its complete result and checked commit
are retained in `contracts.log`. No new ignored tests or roadmap edits exist.
The generated ENV change records the current source revision only.

The fresh census observed thermal 0; the corrected holds and empty-grid packet
observed at most 1. The synthetic candidate screen reached 2 and is screening
evidence only; the full-model paired gate stayed at 1 throughout. One-minute
load averages peaked at 36.2 during the model A/B. Raw process snapshots retain
the background workload; no other qwen model was resident before either model
load. Individual diagnostic/model executions were at most 125 s; queued
build-plus-model holds were below 15 minutes. `leases.jsonl` retains acquisition
and wait history.

At `2026-11-12T16:57:18Z`, the M5 worktree, its lane branch, seven transfer
bundles, four standalone build directories and the 1.3 GB Cargo target were
removed. `cleanup.json` verifies no lane bundle or target remains. Frozen build
identities, the model A/B binary SHA256, raw JSONL, logs and runner remain in the
run directory. The local review worktree/branch remains; no merge was performed.

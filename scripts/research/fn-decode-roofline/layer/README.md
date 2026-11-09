# Causal GDN and QSA layer prototypes

Research sources for the next bounded experiment after the decode decomposition.
The committed production library and shaders remain unchanged. Apply only in an
isolated worktree whose production sources match the parent experiment:

```sh
python3 scripts/research/fn-decode-roofline/layer/prepare.py
lockf -k "$HOME/.cache/qwen-llm/locks/build.lock" nice -n 10 \
  cargo build --release -p qwen-llm --example fn_layer_prototype
python3 scripts/research/fn-decode-roofline/layer/run.py \
  --model "$MODEL" --binary target/release/examples/fn_layer_prototype \
  --out "$HOME/qwen-runs/fn-layer-prototype" --mode screen
python3 scripts/research/fn-decode-roofline/layer/summarize.py \
  "$HOME/qwen-runs/fn-layer-prototype/probe.jsonl" \
  --output "$HOME/qwen-runs/fn-layer-prototype/summary.json"
```

The runner takes the canonical Metal lease only for the probe, with wait enabled,
and records thermal pressure every second. The build lock is not held for timing.
`confirm` measures both ordinary and fourth-token publication phases with twelve
repetitions per cell; `screen` measures the ordinary phase with six. Cell order
reverses on alternate repetitions. Every sample is retained.

The actual incumbent 8K graph supplies layers 2 (GDN) and 3 (QSA), at position
8256 after 8192 salted input IDs and 64 warm decode steps. Each layer begins at
its first HC normalization and ends after its FFN HC injection. It includes both
HC reads/combinations, the mixer, dynamic expert selection, all ten selected
experts and the shared expert. Prefix execution supplies real layer inputs;
the causal prototype computes all following intermediates and route IDs itself.
No recorded future outputs replace computation.

For every sample: replay a complete real token to warm the GPU, restore the
layer's actual entry buffers, flush a separate 1 GiB stream, and time one layer
in one command buffer. Restores, allocations, plan construction, the warm token,
the flush and numerical checks are outside the GPU timing. Native and candidate
use identical cold-weight conditions. This is not a repeated cache-hot layer
loop. Isolated command startup and changes in cross-layer cache behavior limit
full-model extrapolation.

## Executor and fusion variants

- **Incumbent serial** executes the original captured layer, in original order.
- **Incumbent DAG** builds RAW, WAR and WAW edges from reflected used buffer
  ranges, retaining original order for every conflicting pair. Topological waves
  use a concurrent encoder and normal Metal buffer barriers between waves.
- **GDN fused recurrence/norm/gate** keeps a head's 128 recurrence outputs in
  threadgroup memory, then normalizes and gates them. Every group reads immutable
  old state and owns disjoint new state/output rows. Partition counts 1/2/4/8
  trade redundant recurrence work/cache reads for occupancy. There is no spin
  barrier or assumption that another threadgroup is resident.
- **QSA fused QK/softmax/PV** stores block logits in 4 KiB threadgroup memory,
  emits block max/denominator/numerators, then merges them in a separate dispatch.
  Split counts 16/32/64/128 trade occupancy, scratch and merge work. Query and
  arithmetic remain F32; KV remains the incumbent F16.
- **Common FFN epilogue/copy elision** combines routed weighted reduction, shared
  accumulation and HC injection without their intermediate writes. Mixer-output
  copies are bypassed by binding the projection output directly to injection.
- **Shared SiLU/down** additionally evaluates the 640 activations in threadgroup
  memory inside the native IQ4_NL down projection. A distinct 10 KiB output avoids
  racing groups that still read gate/up storage. This deliberately tests whether
  avoided scratch traffic pays for repeated SiLU work; it is separately measured.

All learned projections retain native weight formats and F32 activations. Dense
reductions retain hardware dispatch boundaries. Prototypes preallocate their
scratch; no per-token allocation cost is hidden in a production throughput claim.
The extra GDN state input models a ping-pong state implementation. Integration
would require admitting the second state buffer, 108 MiB over 36 GDN layers.

## Floors and numerical checks

Live used bindings give the useful logical floors at exactly **1170 GB/s**:

| Layer | Weights | Persistent state traffic | Useful total | Floor |
| --- | ---: | ---: | ---: | ---: |
| GDN 2 | 98,686,336 B | 6,537,216 B | 105,223,552 B | 89.935 us |
| QSA 3, ordinary phase | 85,291,520 B | 4,726,784 B | 90,018,304 B | 76.939 us |

These are the same useful-payload definition as the parent census: actual mixed
weight types, ten selected expert banks, read/write state and selected KV/index
payload. They omit scratch/cacheline/repeated traffic and are not DRAM counters.
GDN partition count `P` additionally reads `(P-1) * 3,145,728` state bytes and
recomputes recurrence rows; that cache/ALU cost belongs to the excess over the
useful floor. Treat **140/180 us** as design kill gates, not physical ceilings.

Incumbent serial/DAG must match all architectural boundary bytes. Every candidate
checks finite full hyper residuals, route/selector status, architectural state
and cache bytes, and records max absolute/relative L2 differences. Changed
intermediates are intentionally not compared bitwise.

F64 CPU operator references use the actual GDN q/k/v, beta/decay, old state,
normalization weights and gate, or QSA query, selected IDs, F16 KV and gate.
Report incumbent/candidate RMS and max error, including GDN new state. This is
one actual input per phase, not a distribution qualification campaign. Any
candidate selected for integration still needs signed teacher-forced dNLL/KL
with a repeat noise floor; bitwise text is not the acceptance criterion.

## Separate GPU token-loop projection

The probe also times the existing exact greedy argmax over the real full
vocabulary logits, writing its ID directly into the input buffer of the native
embedding gather in the same GPU command. Check the ID against CPU total-order
greedy, eight repetitions of 32 feedback pairs. This tests feedback mechanics and
cost only: fixed logits are reused, and no full autoregressive loop is claimed.
GPU PLE gathers, dynamic position/cache arguments, stopping and stream delivery
remain separate implementation work. Host-removal projections charge this
measured feedback cost but are upper bounds until those remaining costs exist.
Stochastic sampling is not measured by this greedy feedback probe.

The summary anchors extrapolated layer deltas to the previous measured full
token, **11.994 ms GPU + 1.397 ms nonoverlapping host**, and separately reports
perfect encode hiding and GPU-resident feedback bounds. It does not silently
replace the real model with 48 copies of two representative weight/type mixes.

## Verdict: kill this staged design for the 120+ target

Base `8f2dedac` (production runtime `09a3a521`), M5 Ultra, same GSQ-RCO IQ3_S
artifact. Confirmation has twelve samples/cell at each of positions 8256/8259;
weight ordinary/publication medians 3:1. No sample is discarded. A separate
six-sample ordinary-phase screen agrees.

| Layer | Useful byte floor at 1.17 TB/s | Incumbent serial | Best causal executor | On-chip mixer fusion | Kill gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN 2 | 89.935 us | 199.719 us | **184.677 us** | 190.417 us, two partitions | **140 us: fail** |
| QSA 3 | 76.939 us | 326.151 us | **274.219 us** | 285.188 us, 64 splits | **180 us: fail** |

The best executor is the topological schedule plus FFN scalar epilogue/copy
fusion; it retains the incumbent GDN recurrence and QSA attention. It cuts
24 to 20 / 36 to 32 ordinary dispatches (GDN/QSA), and saves **15.042 / 51.932 us**
per representative layer. Pure DAG scheduling is **194.844 / 284.094 us**.
Best execution reaches only **48.7% / 28.1%** of the useful layer roofs. These
are two actual layer/type mixes and two fixed-input positions, not a full-model
implementation or a confidence interval.

On-chip GDN fusion is exact on all tested hyper residual/state/conv bytes and
has the same F64 error as the incumbent, but is slower across all four occupancy
partitions. QSA fusion is slower across all four split counts. The 64-split
variant's F64 RMS error is **1.205x / 1.126x** incumbent at ordinary/publication
positions; no Flash variant satisfies both RMS/max no-worse screens across both
positions. Small hyper differences (~1e-7 relative L2) are recorded, not rejected
for text or bitwise differences. Shared SiLU/down fusion also loses. The evidence
rejects these implementations; it does not establish that every possible
on-chip fusion or native-quant layout redesign is unprofitable.

Every best-executor/native boundary check is exact, including full state/cache
regions, with the same F64 error. MoE READY=1 and QSA selector status=0 are
checked. The initial setup run incorrectly expected zero for MoE READY and
stopped before retained layer rows; its compact failure log is preserved. It did
not produce a performance acceptance sample. No distribution qualification is
claimed for these performance-killed prototypes.

Anchoring extrapolated 36/12 layer deltas to the prior **13.391 ms** native token
predicts **12.226 ms / 81.79 tok/s**, saving **1.165 ms GPU**. Ideal encode hiding
would give **88.00 tok/s**. Exact GPU greedy argmax plus direct embedding gather
costs **60.80 / 60.88 us** (ordinary/publication), giving an optimistic
GPU-resident feedback bound of **91.83 tok/s** for the best executor, **82.96**
for the incumbent. PLE gathers, dynamic arguments, stochastic sampling and
streaming/stopping are still unimplemented in that loop. These bounds cannot
turn this executor into 120 tok/s; a different GPU design is needed.

Stop at the layer verdict. No production optimization/default is shipped.
Release build and all retained timing/finite/status checks pass; best-executor
state boundaries and F64 checks pass. QSA Flash's numerical screen fails as
reported. No full library/distribution campaign is warranted after the kill.
All **480** retained layer samples (96 screen + 384 confirmation), F64 metrics,
thermal logs (25 monitor readings, all zero), manifests and compact failed-setup
logs live under `~/qwen-runs/2026-11-09-fn-layer-prototype/`, mirrored locally.
Source is reproduced from `056b7986` and the commands above. Under the updated
evidence policy, frozen binaries, sealed sources and regenerable capture bulk
were deleted; `RETENTION.json` records the source commit. The subsequent cache
probe keeps the best executor and GPU feedback candidates for integration:
missing 120 tok/s does not reject their projected 81.8/~92 tok/s gains.

# Flash-Next GDN/QSA Named-Stage Protocol

Status: preregistered before temporary instrumentation or acquisition.

## Question

Does any named packed GDN or dense-QSA mechanism have enough representative
N=512 GPU time to justify a bounded optimization source screen?

The implementation gates are fixed by `docs/PERF-ROADMAP.md`:

- GDN layer-5 score: at least `0.663200 ms/layer` in both split captures.
- QSA layer-7 score: at least `0.805946 ms/layer` in both split captures.

A survivor authorizes investigation inside that composite family. It does not
authorize implementation, prove a constituent is hot, or claim cross-layer or
whole-command savings.

## Workload

- Device: Apple M4 Max.
- Model: internal-SSD UD-Q3_K_XL released checkpoint.
- Prompt: tracked natural 512-token fixture, no special tokens.
- Request: start position zero, exactly 512 packed tokens, one generated token,
  one packed command, no scalar tail, and selected packed QSA disabled.
- Representatives: GDN layer 5 and QSA layer 7.

The QSA split is valid only for the exact dense plan: `dense_tokens=512`,
`selected_tokens=0`, with both conditional `visible_blocks` and `gate` stages
present. Any other plan is `INVALID_SCREEN`.

## Temporary Split

Leave ordinary execution and all kernels unchanged. Add sampled-only siblings
following the existing packed-MoE stage profiler.

GDN uses seven existing labels in order:

1. `gdn.projections`
2. `gdn.decay`
3. `gdn.prep`
4. `gdn.norm`
5. `gdn.recurrence`
6. `gdn.gate_norm`
7. `gdn.output`

Dense QSA uses nine existing labels in order:

1. `qsa.index_projection`
2. `qsa.index_state`
3. `qsa.visible_blocks`
4. `qsa.qkv_projections`
5. `qsa.norm_rope`
6. `qsa.cache_scatter`
7. `qsa.attention`
8. `qsa.gate`
9. `qsa.output`

Retain each `block.mixer` parent at depth 1 and emit children at depth 2. The
temporary layout is 18 stages for layer 5, 14 for layer 7, 78 for the complete
command, 156 samples, and 84 spans. The tail occupies stage 77 and samples
154/155. The existing 256-sample capacity remains unchanged.

## Acquisition

Run separate processes in this fixed order with five seconds between them:

`A1 six-stage -> B1 named split -> B2 named split -> A2 six-stage`

Do not add adaptive warmups, retries, or reordered captures. Preserve executable
and source-diff hashes. A uses the ordinary six-stage profiler; B changes only
the representative sampled profile topology.

Every arm must have:

- accepted GPU and wall observers;
- raw timestamp coverage within the established accepted range;
- the expected 128/68 control or 156/84 split sample/span counts;
- exact labels, depths, endpoint layout, and child containment;
- finite nonnegative durations and no missing or duplicate samples;
- the same generated-output digest and successful built-in deterministic replay.

For each split mixer, child time must not exceed its parent. Otherwise the
attribution is invalid.

## Drift And Scoring

For whole-command GPU, layer-5 `block.mixer`, and layer-7 `block.mixer`, require
control drift

`2 * abs(A2 - A1) / (A1 + A2) <= 0.02`.

Interpolate each A control at the corresponding B capture midpoint and report B
parent inflation, but do not distribute that inflation among children.

For each child duration `t`, calculate the conservative heuristic

`screen_score = max(0, t - 0.018625 ms)`.

The debit is the maximum prior duration of a sampled stage that also copied 5
MiB. It is not a measured boundary cost, corrected mechanism time, or lower
bound. It is used only to make the entry screen pessimistic.

A label survives only when its score independently clears the family threshold
in both B1 and B2. Averaging cannot rescue a failure. Uniform-shape planning
projections may be reported as `34 * GDN score` and `12 * QSA score`, explicitly
named layer-5-scaled and layer-7-scaled projections. Do not sum unrelated
families or call those values observed savings.

The composite projection labels contain multiple operations. A survivor only
opens source analysis within that family. Previously closed mechanisms remain
closed unless this screen identifies a new source of leverage.

## Disposition

- No survivors: remove all instrumentation and advance to selected-QSA semantic
  disposition.
- Survivor: remove all instrumentation, record the family and authority limits,
  then source-screen that family before writing optimization code.
- Any acquisition/layout/drift failure: classify `INVALID_SCREEN`, remove the
  instrumentation, and make no optimization decision.

Adversarial protocol review: `01a04f90-c9dd-76f3-b063-8d1d01e266ae`.

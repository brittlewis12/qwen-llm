# DeepSeek V4 Collapsed FP4 Selector KILL

Date: 2026-08-05

Status: collapsed diagnostics execution is structurally qualified; direct
FP4-Q/K selector replacement is `KILL` on deep quality and all-in speed.

## Question

The bounded no-double-score campaign proved that an instrumented FP4-only arm
could omit F16 scoring while consuming the same FP4 IDs. It did not prove that
those IDs remain useful with more than 513-515 visible rows, nor that FP4 alone
beats the ordinary F16 scorer inside the collapsed token schedule.

This campaign asks both questions at the first useful deeper checkpoint. The
predeclared gates retain the earlier vector thresholds: argmax preservation,
cosine at least 0.999999, relative RMS at most 0.001, maximum absolute error at
most 0.01, at most 5% control drift, and at least 0.75 ms GPU and wall saving
against the faster control.

## Implementation

The feature-gated collapsed path retains 512 selector IDs for every layer and a
compact record carrying eligible visibility, preflight failure detail, consumed
source, and execution schedule. The one-command token reuses query, score, and
mask scratch serially. After completion it bulk-reads records, requires all
inactive slices to remain poisoned, validates every active ID list as sorted,
unique, and in range, validates the dispatch ledger, then hashes traces before
callbacks or token commit.

Execution-aware traces distinguish packed, instrumented singleton, and
collapsed singleton schedules. A second payload digest omits only execution, so
identical selection content can be compared without pretending schedules are
the same. A two-layer one-encoder Metal differential uses distinct queries and
visibilities, reproduces both separate instrumented outputs exactly, and proves
the other 41 slices remain untouched.

## Protocol

One current-asset session enables lineage at position zero and advances the
`[35, 201, 200, 34]` pattern once to position 3,070. It captures snapshot v1
before sealing the counterfactual. Position 3,070 is a paired instrumented audit;
positions 3,071-3,078 are unaudited collapsed FP4-only timing tokens. The
lineage-bearing candidate is then destroyed, and two F16 controls restore from
the same pre-seal snapshot. The audit is excluded from timing.

```bash
cargo test --release -p qwen-llm \
  --test deepseek_v4_position_zero_live \
  --features dsv4-diagnostics \
  current_deepseek_v4_fp4_collapsed_position_3070_packet \
  -- --ignored --exact --nocapture
```

The definitive run completed in 102.70 seconds. The packed prefix consumed
98.36 seconds; it was paid once rather than once per arm. `run.log` binds the
command, harness outcome, elapsed time, packet identity, and headline timings at
SHA-256
`9011284d0ad0df7f015d3a4eb9718c50433e3f2c851f0946140847e99eb024fc`.

## Structural Result

All structural gates pass:

- both restored controls are bit-exact across nine logit vectors, the audit
  decision transcript, and audit/final causal state;
- the paired audit records 21 common / 21 F16 / 21 FP4 invocations;
- every collapsed token records 21 common / zero F16 / 21 FP4 invocations;
- all active and inactive completion records validate;
- exactly 189 CSA-layer selections enter the final trace;
- candidate and controls share exact prefix digests at audit and final.

The full 1,404,624-byte packet remains at
`target/dsv4-current-fp4-collapsed-position3070.json` with SHA-256
`39a4e95b8e94b913e3806ac12b5c204ebd0e4b053ce65e6c432e931586062541`.
`summary.json` retains the environment, toolchain, request, memory, acceptance
gates, control/candidate hashes, causal and trace digests, selector topology,
all six timing sample arrays, and disposition at SHA-256
`42060e844f9c40990e29579176bb2377823939e161d1d253a098fa29e3166ef5`.

The executed five-file campaign source hashes to
`ee4c6eb8749ae519bd58619c4c5348fb7bf1e4829621b219a2de3edb18208825`.
Two post-run hygiene edits make the committed candidate hash
`8404a8f118c20fccbecd6c8291a2997c64b8b552f629784647bdfafd81fc1b0e`:
one narrows a test-only cfg to remove a default-build warning, and one rewrites
an evidence loop for strict Clippy. Applying `executed-source.patch` to the
committed source reconstructs the executed bytes; the patch validates with
`git apply --check` and has SHA-256
`4683af3d6ba941e612e3568768df9cf2e6016e41dc52860fcf416d2710a545ff`.

## Quality KILL

The selector premise fails immediately at the paired audit. Zero of 21 masks
are exact. Symmetric differences range from 24 to 102 IDs, or 12-51 reciprocal
exchanges per layer. The first layer already differs by 102 IDs, so this is not
merely downstream cascade.

| Position | Cosine | Relative RMS | Max abs | Argmax |
|---:|---:|---:|---:|---|
| 3,070 | 0.989320 | 0.150572 | 2.8899 | preserved |
| 3,071 | 0.994970 | 0.100310 | 1.3558 | preserved |
| 3,072 | 0.998423 | 0.056506 | 1.1264 | preserved |
| 3,073 | 0.989291 | 0.146850 | 2.6703 | preserved |
| 3,074 | 0.991872 | 0.130328 | 2.2856 | preserved |
| 3,075 | 0.983313 | 0.182029 | 3.1974 | preserved |
| 3,076 | 0.996973 | 0.077840 | 1.6067 | preserved |
| 3,077 | 0.997399 | 0.075869 | 1.2233 | preserved |
| 3,078 | 0.997278 | 0.075753 | 2.0029 | preserved |

Preserved argmax on a synthetic repeating pattern does not rescue vector drift
that misses the frozen thresholds by orders of magnitude. The earlier shallow
campaign is not contradictory: selecting 512 of 513 candidates can differ by
at most one reciprocal exchange, making it structurally unable to expose this
ranking instability.

## Performance KILL

| Arm | GPU median, ms | Wall median, ms |
|---|---:|---:|
| collapsed FP4 candidate | 45.768 | 47.137 |
| F16 control A | 45.453 | 46.518 |
| F16 control B | 44.312 | 45.366 |

Control drift is valid at 2.54% GPU and 2.51% wall. Against the faster control,
the candidate regresses by 1.456 ms GPU and 1.772 ms wall. The prior
no-double-score saving compared FP4-only with a paired arm that deliberately ran
both scorers; it did not establish FP4-only parity with F16-only execution.

## Decision

Keep the transactional sidecar, paired observer, score plans, ledgers, traces,
collapsed completion records, model-free differentials, and ignored live harness
as diagnostics and negative evidence. They do not alter default builds.

Kill direct FP4-Q/K selection under the current F16-cache product contract. Hold
production FP4 selection, paged FP4 K, snapshot v2, adaptive fallback, and more
same-design timing. Reopen only for a materially new guarded or mixed scorer
that first clears the frozen deep real-weight quality gates and demonstrates a
positive all-in timing ceiling.

The next far-context optimization is exact multi-group selection over the
current F16-authoritative scorer. Its first gate is a cheap production-shape
ceiling; no additional 104 GB prefix is authorized until that clears.

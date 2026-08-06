# DeepSeek V4 Exact R2 Lightning Scorer

Status: formal `INCONCLUSIVE_HOLD`; reviewed exact-R2 design rejected and
removed on an independently stable mandatory cell.

## Question

The production F32-query/F16-key Lightning scorer loads the same 64-head
query once for every compressed row. This packet asks whether one simdgroup
can reuse each query load across exactly two independent rows while preserving
the deployed dimension and head reduction order bit-for-bit.

The candidate is diagnostics-only. It has no production caller and cannot be
promoted by this packet. A pass authorizes only a separately frozen
current-asset A/B campaign for this exact reviewed R2 candidate.

## Identity

- Base revision: `775a2abbe07a1b9949102a6a4632aa1bd5d72af6`.
- Hardware: MacBook Pro `Mac16,5`, Apple M4 Max, 128 GB unified memory.
- OS: macOS 15.6.1 (24G90), Darwin arm64.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`.
- Cargo: `cargo 1.97.1 (c980f4866 2026-06-30)`.
- Metal: Apple metal 32023.864 (`metalfe-32023.864`), target
  `air64-apple-darwin24.6.0`.
- Model/GGUF access: none. Inputs are deterministic synthetic values at
  production geometry.

## Source Freeze

- `kernels/deepseek_v4.metal`:
  `4417da270fd4a57cb4bc43a40d9cfbcd5c0adaf436d909b81c18d6d9c40819fd`
- `crates/qwen-llm/src/deepseek_v4_metal.rs`:
  `f9211592e8413ec29dd7bd7819cff5e10c83e2188796725874fe8de8518f19a0`
- Release test executable:
  `43a691cfc2e715f4554edfc929366bb684bb22a06999174547adcd974b5e28d3`
- Binary diff over the two candidate source files:
  `c9bf8e37dbb568815cc718ad6972b31c8a279a90629071f4caa75ae03a25a0f8`
- Archived `candidate.diff`:
  `c9bf8e37dbb568815cc718ad6972b31c8a279a90629071f4caa75ae03a25a0f8`
- Sole `profile.log`:
  `0d3605f592ae8765c11f82222d859359fcd1070ca687a2edcfde21d0ec0f5048`

Focused release exactness tests pass across scalar, deployed cooperative, and
R2 scorers. They cover repeated score bits, selector outputs, nonfinite
fallback, zero-score ties, nonzero tensor offsets, initialized outputs, input
stability, output guards, and capacities/visibility around 1/2/3, 7/8/9, and
15/16/17. Strict release library Clippy passes.

CX session `019fd685-acb3-7890-83c9-192cbea48c6e` reviewed the kernel,
encoder, tests, timing schedule, statistics, and classifier twice. Its final
static verdict is GO.

## Frozen Protocol

At each of 16,384, 65,536, and 262,144 compressed rows:

1. Run scalar, deployed cooperative, R2, and repeated R2 scorers before
   timing. Require all score bits to match exactly.
2. Require scalar, deployed, and R2 selector mask, IDs, count, and status to
   match exactly.
3. Require query, weights, all initialized key bytes, visibility, and their
   named guards to remain unchanged.
4. Warm with `B C C B`, where B is deployed and C is R2.
5. Retain `(B C C B / C B B C) x 3`: exactly 12 serial one-command,
   one-encoder, one-dispatch samples per arm.
6. Recheck both timed score arms against the frozen scalar vector. Recheck all
   common inputs, score guards, and untouched parent-tail canaries.
7. Compute the median as sorted sample index 6 and p95 as sorted index 11,
   the maximum. No sample is filtered.
8. Compute first-six versus last-six drift as `(max - min) / max`; both arms
   must remain within 5%.

Metal command GPU intervals alone control the gate. Wall intervals are
captured for diagnostics and never control classification:
`wall_is_diagnostic=true`.

Performance gates:

- 16,384 rows: candidate median and p95 regression are each at most 0.05 ms.
- 65,536 rows: median saving is at least 0.15 ms and candidate p95 is below
  baseline median.
- 262,144 rows: median saving is at least 0.50 ms, candidate median is at most
  1.60 ms, and candidate p95 is below baseline median.

Identity, safety, timestamp, or stationarity failure is
`INCONCLUSIVE_HOLD`. A stable performance miss is
`KILL_R2_EXACT_ROW_GROUPING`; R4 is not an automatic rescue. Only a complete
pass emits
`AUTHORIZE_SEPARATELY_FROZEN_CURRENT_ASSET_AB_FOR_REVIEWED_R2_ONLY`.

No-device and unexpected-panic paths fail closed. Ordinary HOLD/KILL verdicts
are emitted once and the test fails outside the guarded gate body.

## Sole Command

```bash
set -o pipefail && cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_lightning_exact_r2_at_far_context \
  -- --ignored --exact --nocapture --test-threads=1 \
  2>&1 | tee \
  "docs/bench/2026-08-06-dsv4-exact-r2-lightning-scorer/profile.log"
```

## Result

| Rows | Baseline median | R2 median | Median delta | Baseline p95 | R2 p95 | Baseline drift | R2 drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 0.658250 ms | 1.127167 ms | +0.468917 ms | 0.669750 ms | 1.134417 ms | 0.9038% | 0.1811% |
| 65,536 | 1.055375 ms | 1.732750 ms | +0.677375 ms | 1.365333 ms | 2.112875 ms | 13.8713% | 13.8560% |
| 262,144 | not run | not run | not run | not run | not run | not run | not run |

All preflight and post-timing score, selector-preflight, input, guard, and
tail-canary checks passed at the executed depths.

The formal packet result is `INCONCLUSIVE_HOLD`. At 65,536 rows, both arms'
half drift exceeded the frozen 5% limit. Its medians are not
classification-grade, the terminal cell was not executed, and this packet
does not authorize a current-asset A/B.

The mandatory 16,384-row cell is independently stable and decisively adverse.
Every R2 sample exceeded every baseline sample. R2 regressed by 0.468917 ms at
the median and failed both frozen no-regression gates. No result at a deeper
cell could repair that conjunctive campaign failure.

The 65,536-row raw samples point in the same direction--every R2 sample also
exceeded every baseline sample--but instability prevents using its medians or
effect size as promotion evidence.

## Decision

Preserve the formal HOLD, but reject and remove this exact reviewed R2 design.
Do not rerun it and do not treat R4 as an automatic rescue. The complete
candidate diff remains in this packet; deployed source is restored.

Reopen multi-row exact scoring only with materially new evidence: compiler or
occupancy attribution that identifies a specific avoidable R2 pathology, or a
structurally new schedule with an analytical explanation for overcoming the
observed 0.468917 ms shallow penalty. Merely changing the row factor, retuning
geometry, or repeating this timing is insufficient.

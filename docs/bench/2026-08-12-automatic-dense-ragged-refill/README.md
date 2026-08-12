# Automatic Dense Ragged Refill

Date: 2026-08-12

Status: promotion candidate; repaired final packet pending fresh audit.

## Question

Should `--execution-mode auto` compose heterogeneous prompt-frontier planning
with the existing bounded two-wave dense B8 refill executor when generation
limits are also heterogeneous?

The executor already preserved one exact token/KV frontier per lane, replaced
finished lanes synchronously, and published input-ordered JSONL. The missing
piece was automatic policy: charged ragged admission required one equal
requested limit, while refill earned its value from skewed limits and remained
explicit under `QWEN_DENSE_BATCH8_REFILL=1`.

## Contract

The automatic joint policy is distinct from broad explicit `Forced` refill. It
requires all of the following:

- dense Qwen B8, greedy regular-file JSONL, and a fixed prefill chunk;
- at least 32 requests forming at least two complete 16-request arenas;
- every prompt at most 256 tokens and covered by one configured prefill chunk;
- every requested generation limit at most 40 tokens and at least two distinct
  limits in the file;
- the equal-frontier planner has zero B8 cohorts and leaves every request serial;
- the candidate covers the entire file with refill arenas: no static cohort or
  serial remainder;
- every arena has at least `9/10` simulated transition utilization and at least
  `3/20` idealized two-wave physical-step savings;
- each arena charges at most 32 prompt tokens per productive transition, while
  the complete admitted file charges at most nine;
- selector and runtime memory admission both pass.

The charge is deliberately file-wide because selection, publication, and the
all-or-nothing fallback transaction are file-wide, and prompt prefill is
additive across arenas. The measured 256-token boundary would be rejected by a
local nine-token charge: its shallow arena charges `1766/56 = 31.54`, its deep
arena `1766/346 = 5.10`, and the complete file `3532/402 = 8.79`. That complete
file nevertheless improves median whole wall `2.35 -> 1.99 s` (`1.181x`) exactly. The
32x local cap bounds subsidy without discarding that measured authority.

Either `QWEN_FIXED_COHORT_RAGGED_PROMPTS=0` or
`QWEN_DENSE_BATCH8_REFILL=0` disables joint admission. Literal refill `=1`
outside automatic mode retains the broader explicit experiment. Qwen MoE and
DeepSeek do not receive dense refill policy fields or execution.

## Final Commands

Candidate and rollback use one final-source binary. `MODEL` is replaced by each
of the two dense anchors; `FIXTURE` is the previously retained
`../2026-08-12-qwen-refill-charged-screen/requests.jsonl`.

```bash
/usr/bin/time -l env \
  -u QWEN_FIXED_COHORT_RAGGED_PROMPTS \
  -u QWEN_DENSE_BATCH8_REFILL \
  target/release/qwen \
  --model MODEL \
  --requests-jsonl FIXTURE \
  --execution-mode auto \
  --temp 0 \
  --prefill-chunk 256
```

Rollback adds `QWEN_DENSE_BATCH8_REFILL=0`. The counterbalanced order is
candidate A, rollback A, rollback B, candidate B.

## Final-Source Result

| Model | Candidate | Rollback | Median speedup | Paired speedups |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B Q8 | `1.78/1.78 s` | `2.14/2.15 s` | `1.205x` | `1.202x/1.208x` |
| Qwen3.6 27B Q4 | `19.12/19.71 s` | `27.91/28.19 s` | `1.445x` | `1.460x/1.430x` |

The claim is median whole-process qualification over this bounded workload,
not universal scheduler authority. Both final-source paired comparisons clear
`1.10x` on both dense anchors.

Candidate selector v3 reports `automatic_dense_refill_admitted`, two planned
arenas, and 32 requests. Dense planner v9 reports the full automatic envelope,
two planned and realized arenas, zero denied/fallback arenas, and transaction
outcome `admitted`. Rollback selects B2 and reports
`automatic_dense_refill_disabled`.

Complete JSONL is byte-identical within each model across all four runs:

- 0.8B SHA-256:
  `e070a28a04c8296564d3589013dbdc3341c4c23ca09612e44394aa2c5eb884ed`
- 27B SHA-256:
  `20a5c39e81093936c1966b524b0e698da4d6ee66de300e95d03a4199b1e49e7d`

Every process reports zero swaps.

## Boundary Evidence

The retained `requests-boundary.jsonl` reaches the 256-token prompt boundary.
On 0.8B, explicit B2-to-refill median wall moves `2.35 -> 1.99 s` (`1.181x`) with
exact output. Its SHA-256 is
`2c278e34d941accfece52b1861ce85292558bc303004310920969a411d5d66d7`.

A qualifying one-arena trace at file charge `8.16` moves only median
`0.86 -> 0.775 s` (`1.110x`), with paired ratios `1.117x/1.103x`. That boundary motivated the minimum two
arenas rather than extrapolating a thin setup-scale gain. Its retained fixture
SHA-256 is
`6fd32a13d0d77721e4f82422ef352cc11deb5cc3c4f023c17a494694d9ed55`.

## Runtime Safety And Telemetry

Selector lookahead prices the candidate maximum shared capacity before choosing
B8. Runtime admission checks the automatic arenas as one transaction. If any
arena denies, execution restores the captured equal-frontier planner baseline
before mutation; it never falls into an unauthorized static-ragged fragment.
Planner v9 separately reports:

- the number of arenas whose physical memory check denied;
- the number abandoned by the all-or-nothing transaction;
- transaction outcome `admitted` or `planner_baseline_fallback`;
- planned versus realized refill arenas, requests, cohorts, and serial fallback.

Explicit refill retains per-arena fallback semantics; this transaction rule is
specific to automatic joint admission.

## Provenance And Validation

The measured binary SHA-256 is
`614e5a9a2d2ca2112cf9e11a89241fa373b9dcda1efa46dcbca4f07f62a5447e`.
It embeds base commit `ea7c4e1feb6375120bd5d5da780af299aeeb1550`, dirty bit one,
and source-state digest
`git-source-sha256-v2:b892daa7d72737138ba5889a6ac97176ac90cb87b5309867129caac0dcc904ac`.
The dirty state includes the candidate source plus two pre-existing user-owned
untracked documents, so authority is same-binary/source-bound rather than a
portable clean-build claim.

The measured model SHA-256 identities are:

- 0.8B Q8:
  `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`
- 27B Q4:
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`

Validation:

- 171 qwen CLI release tests pass; two known fixture-dependent tests are ignored;
- strict `qwen` clippy passes with `-D warnings`;
- `cargo fmt --check` and `git diff --check` pass;
- selector serialization tests prove refill fields are dense-only;
- policy tests freeze inclusive and over-limit 9x file charge, 32x arena
  charge, 90% utilization, and 15% idealized savings boundaries;
- 31 requests cannot enter the two-arena automatic path;
- a synthetic first-arena-admits, second-arena-denies test restores the complete
  planner baseline;

`validation.json` contains exact source hashes, commands, model identities, all
run walls/resources/output hashes, selector records, planner records, and arena
telemetry. Raw stdout/stderr remain disposable; no large trace is retained.

## Decision

Promote the bounded automatic dense slice. This harvests an already-exact
executor and materially improves heterogeneous offline request files before
undertaking a generic continuous scheduler. Keep synchronous MoE refill killed
at current B16 economics, keep root-aware refill held at its `1.015x` optimistic
ceiling, and retain broader ragged/refill behavior as explicit experiments.

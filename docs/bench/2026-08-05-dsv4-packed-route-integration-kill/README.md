# DeepSeek V4 Packed GPU Route Integration KILL

Status: integrated packed GPU routing is `KILL` under the current arithmetic
contract. The exact test-only route/schedule topology remains qualified
diagnostics. Ordinary packed prefill keeps Rust routing, its 542-allocation
memory plan, and its existing expert-major execution.

## Question

The route/schedule microproof established exact topology and bounded GPU cost,
but it compared route weights to a numerical CPU envelope rather than consuming
them through all 43 layers. This packet asks whether the GPU record can replace
Rust routing without changing packed output/state and whether removing that
host seam helps request wall before grouped GPU consumers exist.

## Integrated Seam

The candidate publishes learned or hash routes and a deterministic
expert/token/original-slot schedule in the existing pre-expert command. A
nonzero session generation owns each publication. After completion, the host:

1. validates route, schedule, aggregate, and payload-signature records;
2. compacts fixed-stride slots into the unchanged expert-major schedule; and
3. executes the existing grouped expert path with the published GPU weights.

The validator rejects stale generations, producer failures, invalid IDs,
duplicate routes, nonfinite or negative weights, malformed counts or slots,
padding drift, and payload-signature drift before expert execution.

The integrated switch was never enabled by default. After this KILL, the
ordinary environment switch was removed and all seven buffers, totaling
134,176 logical bytes, were restricted to `dsv4-diagnostics` builds.

## Fixture

- Base revision: `51329b02bcbb24c35fd89618dcb51f224613a267`
- Device: Apple M4 Max
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Prefix: 140 synthetic real-weight tokens, packed as 128 plus 12
- Token pattern: `[35, 201, 200, 34]`
- Continuation: snapshot restore followed by exact token ID 35
- Arms: Rust control, GPU candidate, GPU schedule with Rust weights, Rust control

The first control performs the process-cold 104 GB first touch and is retained
only as evidence. Candidate, hybrid, and trailing control reuse one residency.

Command:

```text
cargo test -p qwen-llm --lib --release \
  --features dsv4-diagnostics \
  current_asset_packed_gpu_route_kill_packet \
  -- --ignored --nocapture
```

## Integrated Result

The two Rust controls have bit-identical packed logits, final hidden state, and
snapshot-restored continuation logits. Their recorded causal, prefix,
compatibility, and continuation-causal digests also match exactly, as do their
committed tokens.

| Arm | Instrumented packed wall | Route generations | Packed logit SHA-256 | Packed causal digest |
|---|---:|---:|---|---|
| Rust before, cold | 39,241.723 ms | 0 | `688aecb312c9...886e` | `0bc001b6045a...1bb` |
| GPU weights | 4,327.494 ms | 86 | `81cb6874e6ad...94f` | `067c8e5c191c...b47` |
| GPU IDs + Rust weights | 4,371.740 ms | 86 | `688aecb312c9...886e` | `0bc001b6045a...1bb` |
| Rust after | 4,250.437 ms | 0 | `688aecb312c9...886e` | `0bc001b6045a...1bb` |

The final timer includes the same-input Rust audit in both GPU arms. Its
77.057 ms candidate/control difference is therefore not an isolated seam cost.
The earlier uninstrumented integration run measures candidate/trailing-control
wall at 4,367.370/4,275.460 ms, a 91.910 ms or 2.15% regression in one warm
comparison. The separate audit run also regresses. Together these are supporting
negative evidence, not a precise all-GPU ceiling: the host validator and
unchanged expert consumer remain in the candidate.

| Observation | Cosine | Relative RMS | Argmax |
|---|---:|---:|---:|
| Packed logits | 0.999258146 | 0.043026507 | 35 / 35 |
| Final hidden | 0.999161524 | 0.041169933 | n/a |
| Restored continuation logits | 0.999561941 | 0.030833979 | 201 / 201 |

Argmax preservation does not satisfy the packed engine's causal-state or
numerical contract. The candidate changes both packed and continuation causal
digests.

## Same-Input Audit

Every candidate layer/chunk recomputes the Rust route from the exact router
inputs consumed by the GPU route. This separates an immediate cutoff mismatch
from drift caused by earlier changed expert outputs.

| Arm | Records | ID mismatch tokens | Symmetric difference | Weight-bit mismatches | Max weight delta | Min learned 6/7 margin |
|---|---:|---:|---:|---:|---:|---:|
| GPU weights | 86 | 0 | 0 | 23,465 | 1.7762e-5 | 2.861e-6 |
| GPU IDs + Rust weights | 86 | 0 | 0 | 23,477 | 2.1335e-5 | 9.54e-7 |

All 86 route ID sets and orderings match for both integrated arms, including
the three hash layers. The current GPU `sqrt(softplus)` and normalization
lineage nevertheless changes many represented route weights by small amounts.
Those deltas are repeatedly amplified by 43 routed residual updates.

The controlled hybrid is decisive: it retains GPU IDs, generation authority,
and the GPU-owned schedule, then overwrites only weights with the same-input
Rust values before unchanged expert execution. Packed logits, final hidden
state, and restored continuation logits become bit-identical to both controls;
packed causal/prefix/compatibility and continuation-causal digests match, as do
committed tokens. Schedule topology is not the cause; route-weight lineage is.

## Decision

Kill production integration of the current GPU route arithmetic. Do not weaken
the packed numerical or state contract to admit a seam that is also slower.

Retain the Metal kernels, deterministic schedule, generation records,
validator/signature, fault harness, and exact N=1..128 microproof under
diagnostics. Reopen only if either:

- GPU weights reproduce the Rust packed route contract exactly; or
- a grouped GPU expert consumer creates enough end-to-end value to justify a
  separately reviewed numerical contract and quality campaign.

The next optimization is the independent grouped IQ2_XS gate/up plus IQ3_XXS
down falsifier at N=12/32/128, fed by the current exact Rust route schedule. It
must clear at least 15% aggregate GPU and wall saving at N=128 before widening
formats or revisiting GPU route ownership.

## Evidence

- Final hybrid/audit log, local path:
  `target/profiles/dsv4-packed-route-integration/hybrid.log`
- Final hybrid/audit log SHA-256:
  `c564337db928a702fd5f591c0dbf951279db0613c331fcb70dcbce82b2e37c99`
- Earlier same-input audit, local path:
  `target/profiles/dsv4-packed-route-integration/audit.log`
- Earlier same-input audit SHA-256:
  `fdc13cdd3a846054c8cfc340f8f56dd11cc398fede4ca863659d7b67ba156d8e`
- Uninstrumented integrated bracket, local path:
  `target/profiles/dsv4-packed-route-integration/first.log`
- Uninstrumented integrated bracket SHA-256:
  `255762d75242b5901b0f55ac14dc6e7ba633c00176daa2bbd3adbe709bbae6e2`
- Final process: 53.89 seconds, zero swaps

The executed uncommitted source and executable hashes were not captured before
the runs and cannot be reconstructed byte-for-byte. The logs therefore bind
the observed outputs, not an archived binary. After the final model run, source
changes are limited to removing the ordinary environment switch,
diagnostics-gating its scratch and encoders, restoring ordinary memory totals,
renaming this ignored harness from a promotion gate to a KILL packet, replacing
its relaxed quality bounds with explicit non-identity/hash assertions, a
Clippy-directed iterator rewrite, and documentation. No route, schedule,
expert, model-math, or hybrid arithmetic changed. Final reviewed source hashes
after formatting and validation are:

- `prefill.rs`:
  `63a05d5407b53243ed9ec00821a1f69aa6fb1a8982bb77c3e186af3859acc3ae`
- `deepseek_v4_metal.rs`:
  `4f1e93506426f45ce812ffa79931425d0effb0ce375340af8d66ed41f3c08c6a`

CX review session: `019fcf7d-e9d4-7150-b496-e70a31958e80`.

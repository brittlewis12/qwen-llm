# Qwen MoE B=16 Product Integration

Update (2026-08-11): prefix-aware packing now clusters complete reusable B=16
groups before depth fallback. See
`docs/bench/2026-08-11-fixed-cohort-prefix-packing/README.md`.

Update (2026-08-11): fixed cohorts now admit heterogeneous per-request generation
limits under the utilization gate documented in
`docs/bench/2026-08-11-fixed-cohort-mixed-limits/README.md`. The original
same-limit qualification below remains unchanged.

Date: 2026-08-11

Status: `GO` for the qualified Qwen 35B-A3B fixed-cohort JSONL path. The product
executor preserves serial output exactly, retains a practical aggregate decode
lead, composes with prefix fanout, and falls back to serial work for incomplete
or incompatible cohorts.

## Scope

The integration starts from committed parent `7707420`, whose benchmark packet
authorized product work after three exact processes reached median `150.300`
aggregate token/s. This packet validates the resulting ordinary `qwen` surface.
The timing rows below come from the final dirty integration tree; they establish
product economics but do not replace the committed benchmark's clean-identity
performance authority.

The backend is deliberately narrow:

- `qwen35moe`, width exactly 16;
- the frozen 40-layer Qwen 35B-A3B architecture and numerical metadata;
- native Q8_0 token embedding and GDN qkv/z/out projections;
- Q4_K routed gate/up banks and a Q6_K language-model head;
- qualified F16 KV storage;
- greedy generation and fixed prompt chunks.

The CLI policy is shared at compile time with dense B=8, while both GPU executors
remain specialized. `--batch-size 8` still requires dense `qwen35`; batch size 16
requires `qwen35moe`. No runtime-polymorphic kernel path was introduced.

## Execution Contract

For each compatible 16-request cohort, the adapter:

1. allocates independent sequence state for every lane;
2. optionally prefills one common prefix and restores it into the other lanes;
3. evaluates private suffixes serially and selects each first pending token with
   the ordinary sampler;
4. consumes subsequent pending tokens in exact B=16 transitions;
5. feeds token zero only to already-finished lanes while active lanes continue;
6. emits ordinary request JSON in original input order.

Compatibility is exact prompt-token count plus requested generation count. A
partial cohort is never padded with fake requests; every remainder uses ordinary
serial inference. Prompt lookup, mutable prefix caches, automatic prompt chunks,
sampling, sidecars, stdin cohorts, and request tracing remain fail-closed.

Capability validation precedes the executor arena and checks architecture,
weight shapes/dtypes, diagnostic graph flags, and queue residency. Every step
also checks model ownership, positions, capacities, mutable aliases, state
inventories, and buffer contracts. Pre-commit errors, cancellation, or panics
restore attention frontiers. A committed command/readback failure poisons the
executor and all cohort sessions. Compute and blit encoders now both finalize on
unwind.

## Product Results

Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

All runs were serialized under the process-wide Metal lease. Every comparison
uses one release binary and one JSONL file for both serial and B=16 execution.

| Fixture | Serial wall | B=16 wall | B=16 decode | Aggregate | Result |
|---|---:|---:|---:|---:|---|
| 16 x 24 prompt / 32 output | 9.437 s | 7.663 s | 3.290 s | 155.614 tok/s | exact |
| 16 x 443 shared prompt / 4 output | 7.898 s | 3.399 s | 0.133 s | 240.793 tok/s | exact |
| 16 x 11 prompt / 4 output | 7.877 s | 4.434 s | 0.322 s | 198.455 tok/s | exact |
| 16 compatible + 2 serial remainders | 4.860 s | 4.682 s | 0.321 s | 199.411 tok/s | exact |

The first row is the charged decode endpoint: 512 generated tokens, 496 logical
transitions, 31 physical B=16 steps, and no padding. Whole-process wall improves
`1.231x` despite serial per-lane prefill and model loading on both sides. Its
complete stdout is byte-identical with SHA-256
`b71b041f01db39b4c0bbc44438641a6826d04909867b16523753ed3de0fae116`.

The shared-prefix row prefills 443 tokens once in `358.650 ms`, snapshots in
`25.839 ms`, restores all other lanes in `52.322 ms`, and completes cohort
prefill in `437.705 ms`. Serial and B=16 output are byte-identical. The model
selects EOS after one B=16 transition, so its aggregate decode rate is a short
window; the meaningful endpoint is the `2.324x` whole-process reduction from
prefix reuse plus exact cohort decode.

The mixed fixture forms one full cohort and two serial fallbacks across three
compatibility buckets. All 18 output objects remain byte-identical and in input
order. Separate CPU tests cover exact 8/16 cohort construction, heterogeneous
bucketing, underfill, prefix alignment and rollback parsing, output reordering,
and productive-versus-padding accounting.

## Memory

The executor owns `17,597,056` bytes of persistent scratch. It is allocated
before cohort admission and is therefore already charged in Metal's current
allocation and memory signals; telemetry reports zero incremental scratch to
avoid double counting.

The 32-token fixture measures `85,704,704` bytes for its first sequence and
admits `1,285,570,560` bytes for the remaining 15, plus the fixed 2 GiB transient
reserve. The 443-token fanout additionally admits a `74,938,172`-byte snapshot.
If only the snapshot makes admission fail, fanout disables and the cohort keeps
ordinary per-lane prefill. A denied sequence cohort fails loudly.

## Decision

Promote Qwen MoE fixed B=16 as a secondary serving capability. It validates the
same-prefix and decode-heavy queue shapes that motivated the lane, while keeping
serial BS=1 behavior and unsupported MoE assets unchanged. The rollback for
prefix reuse is `QWEN_MOE_BATCH16_PREFIX_FANOUT=0`; selecting no batch size
remains the complete executor rollback.

Do not widen this exact executor by geometry inference. Further batch work should
target independently measured product limits: serial heterogeneous prefill,
continuous cohort formation, and broader exact quant coverage. Preserve the
fixed executor's capability gate until each added profile clears equivalent
state and output checks.

## Validation

```text
cargo fmt --all
cargo test -p qwen-llm --lib moe_batch16
cargo test -p qwen-cli --bin qwen fixed_cohort_jsonl
cargo check -p qwen-cli
cargo clippy -p qwen-llm --lib -- -D warnings
cargo clippy -p qwen-cli --all-targets -- -D warnings
cargo build --release -p qwen-cli --bin qwen
git diff --check

./target/release/qwen \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --requests-jsonl FIXTURE.jsonl --temp 0 --prefill-chunk 2048

./target/release/qwen \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --requests-jsonl FIXTURE.jsonl --temp 0 --prefill-chunk 2048 \
  --batch-size 16
```

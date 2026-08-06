# DeepSeek V4 Packed Grouped-IQ2 Phase Ceiling

Status: `INCONCLUSIVE - timestamp-invalid`; the grouped-IQ2 phase lane remains
on `HOLD - closed inconclusive`. The sole frozen campaign stops before any
retained phase cell because an empty gate pre-bracket command has equal Metal
GPU start and end timestamps. No gate or down timing claim survives.

## Question

The current-asset route and count censuses bind the exact N=128 production
schedule for the 25 `IQ2_XS/IQ2_XS/IQ3_XXS` grouped layers. Their accepted
complete routed-stage envelope averages 597.107 ms, but does not assign time
between the deployed gate/up/SwiGLU and down/scatter dispatches.

Measure each phase model-free at production geometry before writing a narrower
tile kernel. A conservative phase upper estimate must reach the existing
158.3 ms economic floor before candidate design is authorized.

## Frozen Protocol

The ignored release-diagnostics harness uses:

- exact route payload SHA-256
  `505cb93ff9c3e1557bbad8f27a773e08b0e8c3475fbb4d7096069667c1fbafdd`;
- H=4096, F=2048, E=256, K=6, N=128, and clamp 10;
- the deployed grouped IQ2 gate/up/SwiGLU and IQ3 down/scatter encoders;
- 25 serial per-layer command buffers per retained phase sample;
- 25 disjoint logical banks plus one all-expert warm bank;
- selected-slice first touch for disjoint banks and complete first touch for
  warm banks with finite nonzero quant scales;
- distinct production-sized input, inner, output, and slot buffers per layer;
- one completed untimed IQ2 gate command immediately before every down command;
- one untimed `D/W/W/D` conditioning block, then matched 25-command empty
  pre/post brackets and `(D/W/W/D) x 6`; and
- complete mutation, finiteness, guard, immutable-weight, immutable-input,
  exact-slot, repeat-digest, error, and timestamp validation.

Every retained cell would contain 12 samples with no filtering or replacement.
Nearest-rank p95 is therefore the maximum. Both cells must satisfy:

```text
drift = 2 * (max(samples) - min(samples))
      / (max(samples) + min(samples)) <= 0.05

U_phase = max(disjoint_p95, warm_p95)
        + max(5 * max(disjoint_range, warm_range),
              empty_pre_gpu_ms,
              empty_post_gpu_ms)
```

Any allocation, Metal, validation, timestamp, or stability failure is
`INCONCLUSIVE`. Stable `U_phase < 158.3 ms` closes only the matching phase-only
lane. Stable `U_phase >= 158.3 ms` authorizes candidate design only.

CX statically reviewed the complete harness after release diagnostics
compilation and required three fail-closed preconditions before execution:
release-only operation, mandatory Metal construction, and an enabled production
grouped-expert policy. All three were added before freezing the source.

## Sole Campaign

The exact reviewed source and binary run once. Gate disjoint/warm validation and
the complete untimed conditioning block finish. The gate empty pre-bracket then
reports equal finite start and end timestamps for command index 1:

```text
1482098.413132..1482098.413132
```

The harness rejects the zero-duration sample exactly as preregistered. It emits
no retained gate cell, never allocates or executes the down campaign, and does
not compute drift, p95, `U_phase`, or a phase decision.

| Provenance fact | Value |
|---|---:|
| In-test elapsed | 19.60 s |
| Process elapsed | 19.79 s |
| Maximum RSS | 9,317,089,280 bytes |
| Swaps | 0 |

These values and the equal timestamps describe execution and instrumentation
only. Conditioning, first touch, and completed validation commands are not
retained timing samples.

## Decision

Record `INCONCLUSIVE - timestamp-invalid`; leave the grouped-IQ2 phase lane on
`HOLD - closed inconclusive`. Publish no gate/down latency, drift, p95,
`U_phase`, KILL, candidate authorization, or product claim. Do not write a
grouped-IQ2 replacement kernel from this packet.

Remove the live one-shot profiler after archiving its exact source and log.
Retain the generally useful route/count fixtures and diagnostics helpers; the
retained production and diagnostics source returns exactly to the base revision.

Do not retry this protocol. Reopen only after a separately reviewed
instrumentation change that conservatively upper-bounds timestamp-resolution-
censored empty commands, never by treating equality as free zero overhead. A
matched minimal nonempty-dispatch bracket is a candidate design, not yet an
authorized campaign. Also reopen for relevant device, Metal, compiler, or
accepted production-observer drift. Any new protocol must preserve 25-command
matching, every retained sample, the 5% stability gate, and phase-only claims.

## Validation

- The executed profiler compiles in release diagnostics before its sole run.
- Both retained route/schedule release diagnostics tests pass after removal.
- Strict retained release diagnostics Clippy passes with warnings denied.
- `cargo fmt --all -- --check` and `git diff --check` pass.
- The retained source diff is empty and every archived hash rechecks.

## Provenance

- Base revision: `90b85a834254479ac2d00c47798d5af14c82f977`.
- Device: Apple M4 Max.
- Executed source diff SHA-256:
  `11cb87eb3d1eeb04ad8990a10a0190528fe62cb33007be1ccc742dc80ca68863`.
- Raw log SHA-256:
  `3961d15deac2aa87e51e9fcab37446d97bbf4c8a8f9bd83425e0662ade2c7101`.
- Executed release test binary SHA-256:
  `bcab454496d0d560a23f65e454b7c57cf651c57dd43f7f55b523bafbb857bdae`.
- Final retained release test binary SHA-256:
  `843dcd945dfbea5082165ea8de0e0bc9d4f30e474359a237e31ea71859120901`.
- Retained source diff SHA-256, empty by construction:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Checksum transcript: `checksums.log`.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

Executed command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::\
profile_exact_route_grouped_iq2_production_phases \
  -- --ignored --exact --nocapture --test-threads=1
```

# DeepSeek V4 Packed Fused All-IQ3 Falsifier

Status: `KILL`. The materially new two-dispatch all-IQ3 path is exact but
slower than the retained four-dispatch diagnostic path at both frozen schedule
extremes. No current-asset run, production policy change, or geometry rescue is
authorized.

## Question

Sixteen current-asset layers use IQ3_XXS gate, up, and down banks. Their retained
diagnostic grouped path dispatches mapped gate, mapped up, clamped SwiGLU, and
mapped down separately. This packet asks whether one fused gate/up/SwiGLU
kernel can reduce that chain to two dispatches while preserving the two mapped
IQ3 reduction lineages exactly and exposing a material full-chain ceiling.

This is material implementation drift from the spent four-dispatch promotion
packet, not a retry. The current asset remains out of scope until the model-free
candidate clears every frozen ceiling gate.

## Candidate

The fused kernel retains independent gate and up IQ3 pointers and eight
`simdgroup_float8x8` accumulators for each projection. For every K tile it:

1. stages the F16 input once in the upper 4 KiB of threadgroup memory;
2. stages and accumulates gate IQ3 weights in the lower 4 KiB;
3. reuses that lower region for the up IQ3 weights after a uniform barrier;
4. publishes gate and up F32 matrices through the existing matrix-store layout;
5. reloads those materialized values after device barriers; and
6. applies the deployed gate/up clamps, sigmoid expression, and product.

The existing mapped IQ3 down/scatter remains unchanged. The Rust encoder is
test-only. The otherwise unreachable Metal function is still compiled into the
ordinary metallib, so this packet claims no zero-byte binary impact.

Host preflight proves IQ3_XXS dtype and production geometry for both banks,
`H=2F`, exact adjacent and disjoint gate/up arena halves, valid map shapes and
schedule geometry, a finite positive clamp, ordered serial encoding, 8 KiB
threadgroup memory, a 32-wide execution width, at least 128 threads, and no
inner/arena overlap.

## Correctness

The active reduced-shape differential uses deterministic nonzero banks at
`H=512`, `F=256`, `E=256`, and `K=6`. It covers
`N=1/12/31/32/33/64/128`, one hybrid schedule with hot expert 255 and routes
spread across lower experts, both expert-ID extremes, source rows distinct from
destination slots, gate above its upper clamp, up across both clamp limits,
repeat identity, staged execution, one-command fused-plus-down execution, and
outer guards. Fused gate, up, inner, and final output are bit-identical to the
retained four-dispatch path.

A separate active contract test rejects malformed bank dtype, map dtype,
schedule geometry, inner/arena overlap, and nonpositive clamp before command
submission.

At production shape the timed harness checks final output and inner bits before
and after timing. Gate/up intermediate identity at that shape is not separately
read back; it is established directly by the reduced-shape active matrix. The
sparse dispatch census observes one serial encoder and exactly four control
versus two candidate dispatches. All output guards survive.

## Frozen Protocol

- Geometry: `H=4096`, `F=2048`, `E=256`, `K=6`, `N=128`.
- Fixture: three deterministic IQ3_XXS banks totaling 2,466,250,752 bytes;
  total Metal allocation is 2,468,823,040 bytes.
- Hot schedule: six experts, each receiving all 128 token rows.
- Sparse schedule: all 256 experts, each receiving three routes.
- Setup: allocation, pipeline creation, first touch, readback, and five
  alternating untimed warms per arm remain outside timing.
- Timing: 24 leading-control, 24 candidate, and 24 trailing-control samples per
  schedule; GPU command time and complete command wall are both retained.
- Stationarity: control and candidate-half median drift must each be at most 5%.
- Promotion: candidate GPU median saving at least 15%, wall median saving at
  least 10%, and GPU plus wall p95 saving at least 10% on both schedules.
- Adjudication: exactness/resource failure or any missed threshold kills the
  candidate. No same-layout geometry sweep is a rescue.

Control comparison uses the faster leading/trailing median and the lower
leading/trailing p95. Percentiles use nearest rank. The complete raw vectors are
preserved in `profile.log`.

## Result

The first and only timed execution is stationary and decisively negative:

| Schedule | GPU control drift | Wall control drift | GPU candidate drift | Wall candidate drift | GPU median saving | Wall median saving | GPU p95 saving | Wall p95 saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hot | 0.0119% | 0.1760% | 0.0053% | 0.0310% | -0.5713% | -0.7625% | -0.5682% | -0.7604% |
| Sparse | 0.0037% | 0.0058% | 0.0012% | 0.0195% | -1.8763% | -1.8684% | -1.8890% | -1.8592% |

The candidate misses every performance threshold while all controls are more
than an order of magnitude inside the drift limit. Reducing four dispatches to
two therefore does not offset the fused kernel's cost. Sixteen live matrix
accumulators are a plausible pressure source, but this packet does not claim
hardware attribution.

## Decision

Retain the exact kernel, active differentials, and sealed ignored harness as
negative diagnostics evidence. Do not run the current 104 GB asset, enable a
policy seam, retune this same accumulator geometry, relax thresholds, or repeat
the same condition.

The next force-ranked packed-prefill lane is newly preregistered
precision-recovering Q8 output arithmetic. It must differ materially from the
killed one-pass F16-staged matrix condition and recover activation quality
before earning another asset campaign.

## Provenance

- Base revision: `d52815a39fad3c967f014a1bdebc1288aa87723d`.
- Device: Apple M4 Max.
- OS: macOS 15.6.1 (24G90).
- Rust/Cargo: 1.97.1.
- Executed `prefill.rs` SHA-256:
  `b21b9dba824e4411e910afa96d2bb354564518add6bb4dfeeed6bc6d6f1c6707`.
- Executed `mat_mat_iq3.metal` SHA-256:
  `3fad9f9d07f2dde4cc861cf543cd4c7ea71a7406529ce9b605b2889ad08727df`.
- Executed release test binary SHA-256:
  `96fc63a6c517d88a62cd4a9b6597cac90809b27204772a830b2b29187a585e0b`.
- Raw `profile.log` SHA-256:
  `692bd7e589d73a95fa219b04c3d2893e8d9e160cae8778e1f06c70c3d7f7df0b`.
- Sealed `prefill.rs` SHA-256:
  `e4f911d6a173008e747a00ef26487a528c8f9f6ac9a36aa16cb1199a4cb910d6`.

The executed and sealed Rust source differs only in the ignored-test label,
which now forbids rerunning the sealed KILL absent material implementation or
device drift. CX review session:
`019fcf7d-e9d4-7150-b496-e70a31958e80`.
